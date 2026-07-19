// TCP server for the dronecv protocol inside Unity.
// Accept + per-client reads run on background threads; decoded messages are
// queued and drained on the main thread by SimLoop. Sends happen from the
// main thread under a per-client lock (blocking writes are fine at sim rates).

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace DroneCV.Flight.Protocol
{
    public class BridgeClient
    {
        public readonly TcpClient Tcp;
        public readonly object SendLock = new object();
        public string Role = "localizer";
        public bool HelloDone;
        public readonly HashSet<string> Channels = new HashSet<string>();
        public long MsgCounter;

        public BridgeClient(TcpClient tcp) { Tcp = tcp; }
    }

    public class InboundMessage
    {
        public BridgeClient Client;
        public DecodedMessage Message;
    }

    public class TcpBridge : IDisposable
    {
        private TcpListener _listener;
        private Thread _acceptThread;
        private volatile bool _running;
        private readonly List<BridgeClient> _clients = new List<BridgeClient>();
        private readonly object _clientsLock = new object();
        public readonly ConcurrentQueue<InboundMessage> Inbound = new ConcurrentQueue<InboundMessage>();

        public void Start(int port)
        {
            _listener = new TcpListener(IPAddress.Any, port);
            _listener.Start();
            _running = true;
            _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "dronecv-accept" };
            _acceptThread.Start();
        }

        public List<BridgeClient> Clients
        {
            get { lock (_clientsLock) return new List<BridgeClient>(_clients); }
        }

        private void AcceptLoop()
        {
            while (_running)
            {
                TcpClient tcp;
                try { tcp = _listener.AcceptTcpClient(); }
                catch { break; }
                tcp.NoDelay = true;
                var client = new BridgeClient(tcp);
                lock (_clientsLock) _clients.Add(client);
                var t = new Thread(() => ReadLoop(client)) { IsBackground = true, Name = "dronecv-read" };
                t.Start();
            }
        }

        private void ReadLoop(BridgeClient client)
        {
            try
            {
                var stream = client.Tcp.GetStream();
                while (_running && client.Tcp.Connected)
                {
                    var msg = MessageCodec.Decode(stream);
                    Inbound.Enqueue(new InboundMessage { Client = client, Message = msg });
                }
            }
            catch
            {
                // Disconnect or malformed stream: drop the client.
            }
            finally
            {
                Remove(client);
            }
        }

        public void Send(BridgeClient client, Header msg, List<Blob> blobs = null)
        {
            try
            {
                lock (client.SendLock)
                {
                    msg.MsgId = ++client.MsgCounter;
                    var data = MessageCodec.Encode(msg, blobs);
                    client.Tcp.GetStream().Write(data, 0, data.Length);
                }
            }
            catch
            {
                Remove(client);
            }
        }

        private void Remove(BridgeClient client)
        {
            lock (_clientsLock) _clients.Remove(client);
            try { client.Tcp.Close(); } catch { }
        }

        public void Dispose()
        {
            _running = false;
            try { _listener?.Stop(); } catch { }
            lock (_clientsLock)
            {
                foreach (var c in _clients) { try { c.Tcp.Close(); } catch { } }
                _clients.Clear();
            }
        }
    }
}
