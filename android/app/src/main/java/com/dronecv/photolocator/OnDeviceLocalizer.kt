package com.dronecv.photolocator

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt
import org.json.JSONObject

/**
 * On-device mode: runs the exported ONNX models with ONNX Runtime, using the
 * NNAPI execution provider so the phone's NPU/DSP accelerates inference where
 * available (CPU fallback otherwise).
 *
 * The math here is a line-by-line Kotlin port of
 * `dronecv.localization.single_shot` (preprocess contract, LandmarkDB top-k
 * consensus with similarity gating, inverse-variance fusion, agreement-scaled
 * confidence). Keep the two in sync.
 */
class OnDeviceLocalizer(bundleDir: File) : AutoCloseable {

    // ---- manifest ----
    private val manifest = JSONObject(File(bundleDir, "manifest.json").readText())
    private val lat0 = manifest.getJSONObject("anchor").getDouble("lat0")
    private val lon0 = manifest.getJSONObject("anchor").getDouble("lon0")
    private val alt0 = manifest.getJSONObject("anchor").getDouble("alt0")
    private val posScaleM = manifest.getDouble("pos_scale_m")
    private val aprSigmaScale = manifest.optDouble("apr_sigma_scale", 1.0)
    private val width = manifest.getJSONObject("camera").getInt("width")
    private val height = manifest.getJSONObject("camera").getInt("height")

    // ---- landmark db ----
    private val dbN: Int
    private val dbD: Int
    private val embeddings: FloatArray // N*D
    private val positions: FloatArray  // N*3 (ENU)
    private val headings: FloatArray   // N

    // ---- onnx sessions ----
    private val env = OrtEnvironment.getEnvironment()
    private val embedSession: OrtSession
    private val poseSession: OrtSession

    init {
        val buf = ByteBuffer.wrap(File(bundleDir, "landmark_db.bin").readBytes())
            .order(ByteOrder.LITTLE_ENDIAN)
        require(buf.int == 0x444C4442) { "bad landmark_db magic" }
        require(buf.int == 1) { "unsupported landmark_db version" }
        dbN = buf.int
        dbD = buf.int
        embeddings = FloatArray(dbN * dbD).also { buf.asFloatBuffer().get(it) }
        buf.position(buf.position() + dbN * dbD * 4)
        positions = FloatArray(dbN * 3).also { buf.asFloatBuffer().get(it) }
        buf.position(buf.position() + dbN * 3 * 4)
        headings = FloatArray(dbN).also { buf.asFloatBuffer().get(it) }

        val opts = OrtSession.SessionOptions().apply {
            try {
                addNnapi() // NPU/DSP acceleration; silently unavailable on some devices
            } catch (_: Throwable) {
            }
        }
        embedSession = env.createSession(File(bundleDir, "embed.onnx").absolutePath, opts)
        poseSession = env.createSession(File(bundleDir, "posenet.onnx").absolutePath, opts)
    }

    fun localize(bitmap: Bitmap, targetErrM: Double = 30.0): LocalizationResult {
        val t0 = System.currentTimeMillis()
        val input = preprocess(bitmap)

        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(input), longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val desc: FloatArray
        val posScaled: FloatArray
        val headingVec: FloatArray
        val logvarPos: Float
        tensor.use {
            embedSession.run(mapOf("image" to tensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                desc = (out[0].value as Array<FloatArray>)[0]
            }
            poseSession.run(mapOf("image" to tensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                posScaled = (out[0].value as Array<FloatArray>)[0]
                @Suppress("UNCHECKED_CAST")
                headingVec = (out[1].value as Array<FloatArray>)[0]
                logvarPos = (out[2].value as FloatArray)[0]
            }
        }

        // ---- retrieval consensus (mirror of LandmarkDB.query) ----
        val sims = FloatArray(dbN)
        for (i in 0 until dbN) {
            var s = 0f
            val off = i * dbD
            for (j in 0 until dbD) s += embeddings[off + j] * desc[j]
            sims[i] = s
        }
        val k = minOf(5, dbN)
        val top = sims.indices.sortedByDescending { sims[it] }.take(k)
        val kept = top.filter { sims[it] > sims[top[0]] - 0.08f }
        val maxSim = sims[kept[0]].toDouble()
        val weights = kept.map { exp((sims[it] - maxSim) / 0.02) }
        val wSum = weights.sum()
        val posR = DoubleArray(3)
        for ((idx, i) in kept.withIndex()) {
            for (a in 0 until 3) posR[a] += weights[idx] / wSum * positions[i * 3 + a]
        }
        var spread = 0.0
        for ((idx, i) in kept.withIndex()) {
            var d2 = 0.0
            for (a in 0 until 3) {
                val d = positions[i * 3 + a] - posR[a]
                d2 += d * d
            }
            spread += weights[idx] / wSum * d2
        }
        val sigmaR = max(sqrt(spread + 0.5), 2.0)
        var sy = 0.0
        var cy = 0.0
        for ((idx, i) in kept.withIndex()) {
            val h = Math.toRadians(headings[i].toDouble())
            sy += weights[idx] / wSum * kotlin.math.sin(h)
            cy += weights[idx] / wSum * cos(h)
        }
        val headingDeg = (Math.toDegrees(atan2(sy, cy)) + 360.0) % 360.0

        // ---- APR (mirror of PoseNet.predict + apr measurement scaling) ----
        val posA = DoubleArray(3) { posScaled[it] * posScaleM }
        val sigmaA = max(
            exp(0.5 * logvarPos.toDouble()).let { it * posScaleM / sqrt(3.0) } * aprSigmaScale,
            2.0
        )

        // ---- inverse-variance fusion + agreement (mirror of single_shot) ----
        val wR = 1.0 / (sigmaR * sigmaR)
        val wA = 1.0 / (sigmaA * sigmaA)
        val pos = DoubleArray(3) { (posR[it] * wR + posA[it] * wA) / (wR + wA) }
        val sigmaFused = sqrt(1.0 / (wR + wA))
        val de = posR[0] - posA[0]
        val dn = posR[1] - posA[1]
        val disagreement = sqrt(de * de + dn * dn)
        val sigmaPair = sqrt(sigmaR * sigmaR + sigmaA * sigmaA)
        val agreement = exp(-0.5 * (disagreement / max(sigmaPair, 1e-6)) * (disagreement / max(sigmaPair, 1e-6)))
        val sigmaEff = sigmaFused / max(agreement, 0.05)
        val confidence = ((1.0 - exp(-(targetErrM * targetErrM) / (2.0 * sigmaEff * sigmaEff))) *
            (0.5 + 0.5 * agreement)).coerceIn(0.0, 1.0)

        // ---- ENU -> WGS84 (small-area approximation, fine at km scale) ----
        val lat = lat0 + pos[1] / 111_320.0
        val lon = lon0 + pos[0] / (111_320.0 * cos(Math.toRadians(lat0)))
        val alt = alt0 + pos[2]

        return LocalizationResult(
            lat = lat,
            lon = lon,
            altMsl = alt,
            headingDeg = headingDeg,
            confidence = confidence,
            sigmaHM = sigmaEff,
            source = "on-device",
            inferenceMs = System.currentTimeMillis() - t0,
        )
    }

    /**
     * Preprocess contract shared with dronecv.localization.single_shot:
     * center-crop to the model aspect ratio, bilinear resize, RGB [0,1],
     * CHW layout.
     */
    private fun preprocess(src: Bitmap): FloatArray {
        val targetAr = width.toFloat() / height
        val ar = src.width.toFloat() / src.height
        val cropped = when {
            ar > targetAr -> {
                val newW = (src.height * targetAr).toInt()
                Bitmap.createBitmap(src, (src.width - newW) / 2, 0, newW, src.height)
            }
            ar < targetAr -> {
                val newH = (src.width / targetAr).toInt()
                Bitmap.createBitmap(src, 0, (src.height - newH) / 2, src.width, newH)
            }
            else -> src
        }
        val scaled = Bitmap.createScaledBitmap(cropped, width, height, true)
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        val out = FloatArray(3 * width * height)
        val plane = width * height
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = ((p shr 16) and 0xFF) / 255f            // R
            out[plane + i] = ((p shr 8) and 0xFF) / 255f     // G
            out[2 * plane + i] = (p and 0xFF) / 255f         // B
        }
        return out
    }

    override fun close() {
        embedSession.close()
        poseSession.close()
    }

    companion object {
        fun isAvailable(bundleDir: File): Boolean =
            File(bundleDir, "manifest.json").exists() &&
                File(bundleDir, "embed.onnx").exists() &&
                File(bundleDir, "posenet.onnx").exists() &&
                File(bundleDir, "landmark_db.bin").exists()
    }
}
