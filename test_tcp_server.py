#!/usr/bin/env python3
"""
Test script to verify TCP server is working correctly
"""
import asyncio
import json
import struct
import base64
import numpy as np
import sys
import time

async def test_tcp_connection():
    """Test TCP server connection and basic functionality"""
    host = 'localhost'
    port = 8766
    
    print(f"🧪 Testing TCP server on {host}:{port}")
    
    try:
        # Connect to TCP server
        reader, writer = await asyncio.open_connection(host, port)
        print("✅ Connected to TCP server")
        
        # Read initial connection message
        length_data = await reader.readexactly(4)
        message_length = struct.unpack('!I', length_data)[0]
        message_data = await reader.readexactly(message_length)
        response = json.loads(message_data.decode('utf-8'))
        
        print(f"✅ Received connection message: {response['type']}")
        print(f"   Message: {response.get('message', 'N/A')}")
        
        # Test 1: Send ping
        print("\n📍 Test 1: Sending ping...")
        ping_msg = json.dumps({"type": "ping"})
        ping_bytes = ping_msg.encode('utf-8')
        writer.write(struct.pack('!I', len(ping_bytes)) + ping_bytes)
        await writer.drain()
        
        # Read pong response
        length_data = await reader.readexactly(4)
        message_length = struct.unpack('!I', length_data)[0]
        message_data = await reader.readexactly(message_length)
        pong = json.loads(message_data.decode('utf-8'))
        
        if pong['type'] == 'pong':
            print("✅ Received pong response")
        else:
            print(f"❌ Unexpected response: {pong['type']}")
        
        # Test 2: Send status request
        print("\n📍 Test 2: Requesting server status...")
        status_msg = json.dumps({"type": "status"})
        status_bytes = status_msg.encode('utf-8')
        writer.write(struct.pack('!I', len(status_bytes)) + status_bytes)
        await writer.drain()
        
        # Read status response
        length_data = await reader.readexactly(4)
        message_length = struct.unpack('!I', length_data)[0]
        message_data = await reader.readexactly(message_length)
        status = json.loads(message_data.decode('utf-8'))
        
        if status['type'] == 'status':
            print("✅ Received status response")
            print(f"   Model info: {status.get('model_info', {}).get('status', 'unknown')}")
            print(f"   Connected clients: {status.get('connected_clients', 0)}")
        else:
            print(f"❌ Unexpected response: {status['type']}")
        
        # Test 3: Send test audio (silence)
        print("\n📍 Test 3: Sending test audio (silence)...")
        
        # Generate silent audio
        silent_audio = np.zeros(16000, dtype=np.float32) + 0.0001
        audio_b64 = base64.b64encode(silent_audio.tobytes()).decode('utf-8')
        
        audio_msg = json.dumps({
            "type": "audio",
            "audio_data": audio_b64
        })
        audio_bytes = audio_msg.encode('utf-8')
        writer.write(struct.pack('!I', len(audio_bytes)) + audio_bytes)
        await writer.drain()
        
        print("   Waiting for response (may take time for model initialization)...")
        
        # Read audio response with timeout
        try:
            length_data = await asyncio.wait_for(reader.readexactly(4), timeout=60)
            message_length = struct.unpack('!I', length_data)[0]
            message_data = await reader.readexactly(message_length)
            audio_response = json.loads(message_data.decode('utf-8'))
            
            if audio_response['type'] == 'response':
                print("✅ Received audio response")
                print(f"   Text: '{audio_response.get('text', '')}' (empty expected for silence)")
                print(f"   Processing time: {audio_response.get('processing_time_ms', 0):.1f}ms")
                print(f"   VAD filtered: {audio_response.get('filtered_by_vad', False)}")
            elif audio_response['type'] == 'error':
                print(f"❌ Server error: {audio_response.get('message', 'Unknown error')}")
            else:
                print(f"❌ Unexpected response: {audio_response['type']}")
        except asyncio.TimeoutError:
            print("❌ Timeout waiting for audio response (60s)")
        
        # Close connection
        print("\n🔌 Closing connection...")
        writer.close()
        await writer.wait_closed()
        print("✅ Connection closed successfully")
        
        print("\n🎉 All TCP server tests completed!")
        return True
        
    except ConnectionRefusedError:
        print(f"❌ Connection refused - TCP server not running on {host}:{port}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("="*50)
    print("TCP Server Test Suite")
    print("="*50)
    
    # Wait a bit for server to be ready
    print("⏳ Waiting 2 seconds for server readiness...")
    await asyncio.sleep(2)
    
    success = await test_tcp_connection()
    
    if success:
        print("\n✅ TCP server is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ TCP server tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
