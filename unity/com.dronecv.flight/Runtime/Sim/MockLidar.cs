// Mock downward lidar: Physics.Raycast + the same noise model as the
// headless sim (gaussian noise, max range, dropout probability).

using System;
using UnityEngine;

namespace DroneCV.Flight.Sim
{
    public class MockLidar : MonoBehaviour
    {
        public float MaxRange = 120f;
        public float NoiseSigma = 0.05f;
        [Range(0f, 1f)] public float DropoutProb = 0.01f;

        private System.Random _rng = new System.Random(0);

        public void Reseed(int seed) => _rng = new System.Random(seed * 31 + 7);

        /// Range in meters or null (dropout / out of range) — matches
        /// sensors.MockLidar.range_down in the headless sim.
        public double? RangeDown()
        {
            if (_rng.NextDouble() < DropoutProb) return null;
            if (!Physics.Raycast(transform.position, Vector3.down, out var hit, MaxRange))
                return null;
            return Math.Max(0.0, hit.distance + Gaussian() * NoiseSigma);
        }

        private double Gaussian()
        {
            // Box-Muller
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = _rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }
}
