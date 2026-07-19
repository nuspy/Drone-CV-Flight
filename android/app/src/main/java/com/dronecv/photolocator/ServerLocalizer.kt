package com.dronecv.photolocator

import java.io.File
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject

/**
 * Server mode: POSTs the photo to a `dronecv serve` instance
 * (POST /localize) and parses the fix. Also downloads the on-device model
 * bundle (GET /mobile-bundle.zip) so the app can switch to NPU mode.
 */
class ServerLocalizer(private val baseUrl: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    fun localize(photo: File): LocalizationResult {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", photo.name,
                photo.asRequestBody("image/jpeg".toMediaType())
            )
            .build()
        val request = Request.Builder().url("${baseUrl.trimEnd('/')}/localize").post(body).build()
        val t0 = System.currentTimeMillis()
        client.newCall(request).execute().use { resp ->
            if (!resp.isSuccessful) {
                throw RuntimeException("server error ${resp.code}: ${resp.body?.string()}")
            }
            val json = JSONObject(resp.body!!.string())
            return LocalizationResult(
                lat = json.getDouble("lat"),
                lon = json.getDouble("lon"),
                altMsl = json.getDouble("alt_msl"),
                headingDeg = json.getDouble("heading_deg"),
                confidence = json.getDouble("confidence"),
                sigmaHM = json.getDouble("sigma_h_m"),
                source = "server",
                inferenceMs = System.currentTimeMillis() - t0,
            )
        }
    }

    /** Downloads mobile_bundle.zip into [destDir] and unpacks it. */
    fun downloadMobileBundle(destDir: File) {
        val request = Request.Builder().url("${baseUrl.trimEnd('/')}/mobile-bundle.zip").build()
        client.newCall(request).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("bundle download failed: ${resp.code}")
            destDir.mkdirs()
            java.util.zip.ZipInputStream(resp.body!!.byteStream()).use { zin ->
                var entry = zin.nextEntry
                while (entry != null) {
                    val out = File(destDir, File(entry.name).name) // flat, no traversal
                    out.outputStream().use { zin.copyTo(it) }
                    entry = zin.nextEntry
                }
            }
        }
    }
}
