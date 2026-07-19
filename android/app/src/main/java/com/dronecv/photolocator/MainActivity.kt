package com.dronecv.photolocator

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.google.android.gms.maps.model.CameraPosition
import com.google.android.gms.maps.model.LatLng
import com.google.maps.android.compose.GoogleMap
import com.google.maps.android.compose.Marker
import com.google.maps.android.compose.MarkerState
import com.google.maps.android.compose.rememberCameraPositionState
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { LocatorScreen() } }
    }
}

private enum class Mode { SERVER, ON_DEVICE }

@androidx.compose.runtime.Composable
fun LocatorScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var mode by remember { mutableStateOf(Mode.SERVER) }
    var serverUrl by remember { mutableStateOf("http://192.168.1.10:8000") }
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("Take or pick a photo of the trained environment.") }
    var result by remember { mutableStateOf<LocalizationResult?>(null) }

    val photoDir = File(context.cacheDir, "photos").apply { mkdirs() }
    val photoFile = remember { File(photoDir, "photo.jpg") }
    val bundleDir = File(context.filesDir, "mobile_bundle")

    fun localize(file: File) {
        busy = true
        status = "Localizing (${if (mode == Mode.SERVER) "server" else "on-device NPU"})..."
        scope.launch {
            try {
                val fix = withContext(Dispatchers.Default) {
                    when (mode) {
                        Mode.SERVER -> ServerLocalizer(serverUrl).localize(file)
                        Mode.ON_DEVICE -> {
                            if (!OnDeviceLocalizer.isAvailable(bundleDir)) {
                                throw RuntimeException(
                                    "No on-device model. Use 'Download model' (server must be reachable) first."
                                )
                            }
                            val bitmap: Bitmap = BitmapFactory.decodeFile(file.absolutePath)
                                ?: throw RuntimeException("could not decode photo")
                            OnDeviceLocalizer(bundleDir).use { it.localize(bitmap) }
                        }
                    }
                }
                result = fix
                status = "Fix from ${fix.source} in ${fix.inferenceMs} ms"
            } catch (e: Exception) {
                status = "Error: ${e.message}"
            } finally {
                busy = false
            }
        }
    }

    val takePicture = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) localize(photoFile)
    }
    val pickPhoto = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) {
            context.contentResolver.openInputStream(uri)?.use { input ->
                photoFile.outputStream().use { input.copyTo(it) }
            }
            localize(photoFile)
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("DroneCV Photo Locator", style = MaterialTheme.typography.headlineSmall)

        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            SegmentedButton(
                selected = mode == Mode.SERVER,
                onClick = { mode = Mode.SERVER },
                shape = SegmentedButtonDefaults.itemShape(0, 2),
            ) { Text("Server") }
            SegmentedButton(
                selected = mode == Mode.ON_DEVICE,
                onClick = { mode = Mode.ON_DEVICE },
                shape = SegmentedButtonDefaults.itemShape(1, 2),
            ) { Text("On-device (NPU)") }
        }

        OutlinedTextField(
            value = serverUrl,
            onValueChange = { serverUrl = it },
            label = { Text("dronecv serve URL") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !busy, onClick = {
                val uri = FileProvider.getUriForFile(
                    context, "${context.packageName}.fileprovider", photoFile
                )
                takePicture.launch(uri)
            }) { Text("Take photo") }
            Button(enabled = !busy, onClick = {
                pickPhoto.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }) { Text("Pick photo") }
            OutlinedButton(enabled = !busy, onClick = {
                busy = true
                status = "Downloading on-device model from server..."
                scope.launch {
                    try {
                        withContext(Dispatchers.IO) {
                            ServerLocalizer(serverUrl).downloadMobileBundle(bundleDir)
                        }
                        status = "On-device model ready."
                    } catch (e: Exception) {
                        status = "Download failed: ${e.message}"
                    } finally {
                        busy = false
                    }
                }
            }) { Text("Download model") }
        }

        if (busy) CircularProgressIndicator()
        Text(status, style = MaterialTheme.typography.bodyMedium)

        result?.let { fix ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Lat: %.6f   Lon: %.6f".format(fix.lat, fix.lon))
                    Text("Altitude: %.1f m MSL   Heading: %.0f deg".format(fix.altMsl, fix.headingDeg))
                    Text(
                        "Confidence: %.0f%%   (sigma %.0f m, %s)"
                            .format(fix.confidence * 100, fix.sigmaHM, fix.source)
                    )
                }
            }
            val target = LatLng(fix.lat, fix.lon)
            val cameraState = rememberCameraPositionState(fix.hashCode().toString()) {
                position = CameraPosition.fromLatLngZoom(target, 16f)
            }
            Spacer(Modifier.height(4.dp))
            GoogleMap(
                modifier = Modifier.fillMaxWidth().height(320.dp),
                cameraPositionState = cameraState,
            ) {
                Marker(
                    state = MarkerState(position = target),
                    title = "Estimated position",
                    snippet = "±%.0f m, conf %.0f%%".format(fix.sigmaHM, fix.confidence * 100),
                )
            }
        }
    }
}
