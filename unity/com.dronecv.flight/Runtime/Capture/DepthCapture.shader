// Renders eye RANGE (distance from camera position to the fragment, meters)
// into a float render target — the same semantics as the headless
// rasterizer's depth image and the value the mock lidar model expects.
Shader "DroneCV/DepthCapture"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float range = distance(i.worldPos, _WorldSpaceCameraPos);
                return float4(range, 0, 0, 1);
            }
            ENDCG
        }
    }
}
