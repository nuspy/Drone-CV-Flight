// Teleportable camera rig: renders RGB (jpeg/png bytes) and range images at
// arbitrary poses without touching the drone. Depth uses the
// DroneCV/DepthCapture replacement shader into an RFloat target, so the
// returned values are metric ranges identical in meaning to the headless
// rasterizer's depth channel.

using UnityEngine;

namespace DroneCV.Flight.Capture
{
    [RequireComponent(typeof(Camera))]
    public class CaptureRig : MonoBehaviour
    {
        public Shader DepthShader;

        private Camera _cam;

        public Camera Cam
        {
            get
            {
                if (_cam == null) _cam = GetComponent<Camera>();
                return _cam;
            }
        }

        private void Awake()
        {
            if (DepthShader == null) DepthShader = Shader.Find("DroneCV/DepthCapture");
        }

        public void SetPose(Vector3 posSim, float yawDeg, float pitchDownDeg)
        {
            transform.position = posSim;
            transform.rotation = Quaternion.Euler(pitchDownDeg, yawDeg, 0f);
        }

        public byte[] RenderRgb(int width, int height, string format, int jpegQuality)
        {
            var rt = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
            var prevTarget = Cam.targetTexture;
            var prevActive = RenderTexture.active;
            try
            {
                Cam.targetTexture = rt;
                Cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                tex.Apply();
                var bytes = format == "png" ? tex.EncodeToPNG() : tex.EncodeToJPG(jpegQuality);
                Object.Destroy(tex);
                return bytes;
            }
            finally
            {
                Cam.targetTexture = prevTarget;
                RenderTexture.active = prevActive;
                RenderTexture.ReleaseTemporary(rt);
            }
        }

        /// Row-major float array, shape (height, width): range in meters,
        /// 0 where nothing was hit (far plane).
        public float[] RenderDepth(int width, int height)
        {
            var rt = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.RFloat);
            var prevTarget = Cam.targetTexture;
            var prevActive = RenderTexture.active;
            var prevFlags = Cam.clearFlags;
            var prevBg = Cam.backgroundColor;
            try
            {
                Cam.targetTexture = rt;
                Cam.clearFlags = CameraClearFlags.SolidColor;
                Cam.backgroundColor = Color.clear; // range 0 = no hit
                Cam.RenderWithShader(DepthShader, null);
                RenderTexture.active = rt;
                var tex = new Texture2D(width, height, TextureFormat.RFloat, false);
                tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                tex.Apply();
                var raw = tex.GetPixelData<float>(0);
                var result = new float[width * height];
                // Texture rows are bottom-up; the protocol expects top-down.
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        result[y * width + x] = raw[(height - 1 - y) * width + x];
                Object.Destroy(tex);
                return result;
            }
            finally
            {
                Cam.targetTexture = prevTarget;
                Cam.clearFlags = prevFlags;
                Cam.backgroundColor = prevBg;
                RenderTexture.active = prevActive;
                RenderTexture.ReleaseTemporary(rt);
            }
        }
    }
}
