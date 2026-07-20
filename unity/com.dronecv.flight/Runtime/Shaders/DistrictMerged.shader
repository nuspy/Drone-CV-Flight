// One material for a whole merged district: a shared tiling albedo texture
// (the procedural window atlas) multiplied by per-vertex colour, so every
// building's wall/roof tint survives the merge into a single draw material.
// Ships a URP sub-shader and a Built-in RP fallback so it works in both.
Shader "DroneCV/DistrictMerged"
{
    Properties
    {
        _MainTex ("Albedo (tiling window)", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 0.0
    }

    // ---------------- URP ----------------
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" "RenderType" = "Opaque" }
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float2 uv : TEXCOORD0; float4 color : COLOR; };
            struct Varyings { float4 positionHCS : SV_POSITION; float2 uv : TEXCOORD0; float3 normalWS : TEXCOORD1; float4 color : COLOR; };

            TEXTURE2D(_MainTex); SAMPLER(sampler_MainTex);
            float4 _MainTex_ST;

            Varyings vert (Attributes v)
            {
                Varyings o;
                o.positionHCS = TransformObjectToHClip(v.positionOS.xyz);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.normalWS = TransformObjectToWorldNormal(v.normalOS);
                o.color = v.color;
                return o;
            }

            half4 frag (Varyings i) : SV_Target
            {
                half3 albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).rgb * i.color.rgb;
                Light mainLight = GetMainLight();
                half ndl = saturate(dot(normalize(i.normalWS), mainLight.direction));
                half3 ambient = SampleSH(normalize(i.normalWS));
                half3 lit = albedo * (mainLight.color * ndl + ambient + 0.15);
                return half4(lit, 1);
            }
            ENDHLSL
        }
    }

    // ---------------- Built-in RP fallback ----------------
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
