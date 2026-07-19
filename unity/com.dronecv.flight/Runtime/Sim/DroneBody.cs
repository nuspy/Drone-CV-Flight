// Kinematic drone body — the SAME dynamics as the headless sim's drone.py
// (first-order velocity-command lag, clamped speeds, terrain collision), so
// closed-loop controller behavior transfers between simulators.

using UnityEngine;

namespace DroneCV.Flight.Sim
{
    public class DroneBody : MonoBehaviour
    {
        [Header("Limits (mirror configs/*.yaml sim.drone)")]
        public float MaxSpeed = 12f;
        public float MaxClimb = 4f;
        public float MaxYawRateDps = 90f;
        public float ResponseTau = 0.6f;

        [Header("State (read-only)")]
        public Vector3 Velocity;
        public bool Collided;

        private Vector3 _cmdVel;
        private float _cmdYawRate;

        public float YawDeg => transform.eulerAngles.y;

        public void ResetTo(Vector3 pos, float yawDeg)
        {
            transform.position = pos;
            transform.rotation = Quaternion.Euler(0f, yawDeg, 0f);
            Velocity = Vector3.zero;
            _cmdVel = Vector3.zero;
            _cmdYawRate = 0f;
            Collided = false;
        }

        public void SetCommand(Vector3? velSim, float? yawRateDps)
        {
            if (velSim.HasValue)
            {
                var v = velSim.Value;
                var horiz = new Vector3(v.x, 0f, v.z);
                if (horiz.magnitude > MaxSpeed) horiz = horiz.normalized * MaxSpeed;
                var climb = Mathf.Clamp(v.y, -MaxClimb, MaxClimb);
                _cmdVel = new Vector3(horiz.x, climb, horiz.z);
            }
            if (yawRateDps.HasValue)
                _cmdYawRate = Mathf.Clamp(yawRateDps.Value, -MaxYawRateDps, MaxYawRateDps);
        }

        /// Advance by dt using an explicit terrain height probe (raycast).
        public void Step(float dt)
        {
            float alpha = 1f - Mathf.Exp(-dt / Mathf.Max(ResponseTau, 1e-3f));
            Velocity += (_cmdVel - Velocity) * alpha;
            transform.position += Velocity * dt;
            transform.rotation = Quaternion.Euler(0f, YawDeg + _cmdYawRate * dt, 0f);

            var ground = GroundHeightBelow();
            if (ground.HasValue && transform.position.y <= ground.Value + 0.5f)
            {
                var p = transform.position;
                p.y = ground.Value + 0.5f;
                transform.position = p;
                Collided = true;
            }
        }

        public float? GroundHeightBelow()
        {
            var origin = transform.position + Vector3.up * 500f;
            if (Physics.Raycast(origin, Vector3.down, out var hit, 2000f))
                return hit.point.y;
            return null;
        }
    }
}
