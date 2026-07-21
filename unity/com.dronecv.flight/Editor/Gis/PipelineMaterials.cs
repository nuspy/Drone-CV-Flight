// Render-pipeline-aware material creation so the imported scene is never
// magenta: Unity 6 defaults to URP, where `Shader.Find("Standard")` and a
// default Terrain material do not exist. This picks the URP / HDRP / Built-in
// shader for the ACTIVE pipeline and sets the right colour/texture properties.

using UnityEngine;
using UnityEngine.Rendering;

namespace DroneCV.Flight.Editor.Gis
{
    public static class PipelineMaterials
    {
        static bool ScriptableActive => GraphicsSettings.currentRenderPipeline != null;

        /// The lit shader of the active pipeline (URP/HDRP/Built-in).
        public static Shader LitShader()
        {
            if (ScriptableActive)
            {
                var s = Shader.Find("Universal Render Pipeline/Lit");
                if (s != null) return s;
                s = Shader.Find("HDRP/Lit");
                if (s != null) return s;
            }
            return Shader.Find("Standard");
        }

        /// A lit material tinted `color` (and optional albedo texture), valid in
        /// the active pipeline. Double-sided so a native mesh with either winding
        /// still renders.
        public static Material Lit(Color color, Texture tex = null, bool doubleSided = false)
        {
            var mat = new Material(LitShader());
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            if (tex != null)
            {
                if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", tex);
                if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", tex);
            }
            if (doubleSided && mat.HasProperty("_Cull"))
                mat.SetFloat("_Cull", (float)CullMode.Off);
            return mat;
        }

        /// The terrain material for the active pipeline, or null for Built-in
        /// (where Terrain uses its own default material automatically).
        public static Material TerrainMaterial()
        {
            if (!ScriptableActive) return null;
            var s = Shader.Find("Universal Render Pipeline/Terrain/Lit");
            if (s == null) s = Shader.Find("HDRP/TerrainLit");
            return s != null ? new Material(s) : null;
        }
    }
}
