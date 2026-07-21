// One material for a whole merged district: a shared tiling albedo texture
// (the procedural window atlas) multiplied by per-vertex colour, so every
// building's wall/roof tint survives the merge into a single draw material.
//
// Built-in Render Pipeline only (a hand-written URP sub-shader would fail to
// compile in Built-in projects because the URP ShaderLibrary is absent). On
// URP/HDRP, DistrictMerger falls back to the pipeline's Lit shader (one
// material, without the per-vertex tint) — see DistrictMerger.BuildSharedMaterial.
Shader "DroneCV/DistrictMerged"
{
    Properties
    {
        _MainTex ("Albedo (tiling window)", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 0.0
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        CGPROGRAM
        #pragma surface surf Standard vertex:vert
        #pragma target 3.0
        sampler2D _MainTex;
        half _Glossiness;
        struct Input { float2 uv_MainTex; float4 color : COLOR; };
        void vert (inout appdata_full v) { }
        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * IN.color;
            o.Albedo = c.rgb;
            o.Metallic = 0.0;
            o.Smoothness = _Glossiness;
        }
        ENDCG
    }

    Fallback "Diffuse"
}
