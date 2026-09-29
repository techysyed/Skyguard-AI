'use client';

import { useEffect } from 'react';
import { skyguardWebSocket } from '@/lib/live/websocket';

export default function LiveBackendBridge() {
  useEffect(() => {
    console.log('🚀 Starting SkyGuard live backend bridge...');

    skyguardWebSocket.connect();

    const unsubscribe = skyguardWebSocket.subscribe((data) => {
      console.log('📡 LIVE PYTHON DATA RECEIVED:', data);
    });

    return () => {
      unsubscribe();
      skyguardWebSocket.disconnect();
    };
  }, []);

  return null;
}