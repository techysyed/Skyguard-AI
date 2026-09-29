export interface BackendTelemetry {
    timestamp: string;
    station_id: string;
    status: string;
  
    metrics: {
      temperature: number;
      pressure: number;
      humidity: number;
      wind_speed: number;
      anomaly_score: number;
      xai_engine: Record<string, unknown> | unknown;
  
      health: {
        score: number;
        status: string;
        estimated_true_values: {
          temp: number | null;
          pressure: number | null;
          humidity: number | null;
        };
      };
    };
  }
  
  type Listener = (data: BackendTelemetry) => void;
  
  class SkyGuardWebSocket {
    private socket: WebSocket | null = null;
    private listeners = new Set<Listener>();
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  
    connect() {
      if (typeof window === 'undefined') return;
  
      if (
        this.socket &&
        (this.socket.readyState === WebSocket.OPEN ||
          this.socket.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }
  
      this.socket = new WebSocket('ws://127.0.0.1:8000/ws');
  
      this.socket.onopen = () => {
        console.log('🟢 SkyGuard Python backend connected');
      };
  
      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as BackendTelemetry;
  
          console.log(
            `📡 Backend telemetry: ${data.station_id} | ${data.status}`,
          );
  
          this.listeners.forEach((listener) => listener(data));
        } catch (error) {
          console.error('❌ Invalid backend WebSocket message:', error);
        }
      };
  
      this.socket.onclose = () => {
        console.log('🔴 SkyGuard Python backend disconnected');
  
        this.socket = null;
  
        this.reconnectTimer = setTimeout(() => {
          this.connect();
        }, 3000);
      };
  
      this.socket.onerror = (error) => {
        console.error('❌ SkyGuard WebSocket error:', error);
      };
    }
  
    disconnect() {
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
  
      this.socket?.close();
      this.socket = null;
    }
  
    subscribe(listener: Listener) {
      this.listeners.add(listener);
  
      return () => {
        this.listeners.delete(listener);
      };
    }
  }
  
  export const skyguardWebSocket = new SkyGuardWebSocket();