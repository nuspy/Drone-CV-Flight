# DroneCV Photo Locator (Android)

Quick-test companion app for the dronecv system: **take or pick a photo** of
a trained environment (e.g. an aerial picture) and get the estimated
**coordinates, altitude and position on Google Maps**, with the model's
confidence.

Two interchangeable inference modes:

| Mode | How it works | When to use |
|---|---|---|
| **Server** | photo is POSTed to a `dronecv serve` instance (`POST /localize`) | development, big models, no phone setup |
| **On-device (NPU)** | the exported ONNX bundle runs locally with ONNX Runtime + the NNAPI execution provider (routes to the phone's NPU/DSP where available, CPU fallback) | offline field tests, latency |

The on-device math (preprocessing, retrieval consensus, APR fusion,
confidence) is a Kotlin port of `dronecv.localization.single_shot` — keep the
two in sync.

## Build

1. Android Studio (Ladybug+) or plain Gradle with the Android SDK (compileSdk
   35). Open the `android/` folder.
2. Google Maps key: create `android/local.properties` with
   ```
   MAPS_API_KEY=<your Google Maps Android API key>
   ```
   (the map stays blank without it; localization still works).
3. Run on a device (minSdk 27).

## Use

1. Train a model for your environment and start the server on a machine the
   phone can reach:
   ```bash
   dronecv run-all --env <env>          # or: dronecv train --env <env>
   dronecv serve --env <env> --host 0.0.0.0 --port 8000
   ```
2. In the app, set the server URL (e.g. `http://192.168.1.10:8000`).
3. **Server mode**: take/pick a photo → fix + marker on the map.
4. **On-device mode**: tap *Download model* once (fetches
   `/mobile-bundle.zip`, ~a few MB), then switch the toggle — photos are
   localized entirely on the phone. You can also export the bundle manually
   with `dronecv export --env <env>` and push it to
   `files/mobile_bundle/` via `adb`.

## Notes

- Photos must show the environment the model was trained on; anything else
  yields a low-confidence fix (the confidence value is meaningful — trust it).
- The ENU→WGS84 conversion on-device uses a local flat-earth approximation
  (fine at the km scale of a single environment; the server uses exact
  geodesy).
- `usesCleartextTraffic` is enabled for plain-HTTP LAN servers; put the
  server behind HTTPS for anything beyond bench testing.
