// These tests run the server against a fake OpenAI Realtime server, so no API key or credits are needed.

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

// newFakeOpenAI starts a fake Realtime server. Every message it receives is sent to got.
// On input_audio_buffer.append it replies with an audio delta, an error event and speech_started.
func newFakeOpenAI(t *testing.T, got chan<- map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("Authorization header = %q", auth)
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow()

		ctx := r.Context()
		for {
			var m map[string]any
			if err := wsjson.Read(ctx, c, &m); err != nil {
				close(got)
				return
			}
			got <- m
			if m["type"] == "input_audio_buffer.append" {
				wsjson.Write(ctx, c, map[string]any{"type": "response.output_audio.delta", "delta": "QUJD", "item_id": "item_42"})
				wsjson.Write(ctx, c, map[string]any{"type": "error", "error": map[string]any{"code": "x", "message": "test error"}})
				wsjson.Write(ctx, c, map[string]any{"type": "input_audio_buffer.speech_started"})
			}
		}
	}))
}

func wsURL(httpURL string) string { return "ws" + strings.TrimPrefix(httpURL, "http") }

func TestBridge(t *testing.T) {
	got := make(chan map[string]any, 10)
	fake := newFakeOpenAI(t, got)
	defer fake.Close()

	srv := httptest.NewServer(newMux("test-key", wsURL(fake.URL)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Same-origin connection, as the browser makes after loading the page from this server
	browser, _, err := websocket.Dial(ctx, wsURL(srv.URL)+"/voice", &websocket.DialOptions{
		HTTPHeader: http.Header{"Origin": []string{srv.URL}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer browser.CloseNow()

	// 1. The session is initialized first, in GA format
	m := <-got
	if m["type"] != "session.update" {
		t.Fatalf("first message = %v, want session.update", m)
	}
	session := m["session"].(map[string]any)
	output := session["audio"].(map[string]any)["output"].(map[string]any)
	if session["type"] != "realtime" || output["voice"] != "coral" {
		t.Fatalf("unexpected session config: %v", session)
	}

	// 2. Microphone audio is forwarded as input_audio_buffer.append
	wsjson.Write(ctx, browser, clientMessage{Audio: "AAAA"})
	if m := <-got; m["type"] != "input_audio_buffer.append" || m["audio"] != "AAAA" {
		t.Fatalf("append = %v", m)
	}

	// 3. Assistant audio reaches the browser
	var out clientMessage
	if err := wsjson.Read(ctx, browser, &out); err != nil || out.Audio != "QUJD" {
		t.Fatalf("audio to browser = %+v, err = %v", out, err)
	}

	// 4. speech_started truncates the assistant item that was playing
	if m := <-got; m["type"] != "conversation.item.truncate" || m["item_id"] != "item_42" {
		t.Fatalf("truncate = %v", m)
	}

	// 5. When the browser hangs up, the upstream connection is closed too
	browser.Close(websocket.StatusNormalClosure, "")
	select {
	case m, ok := <-got:
		if ok {
			t.Fatalf("unexpected message after hang-up: %v", m)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("upstream connection was not closed after the browser hung up")
	}
}

func TestRejectsCrossOrigin(t *testing.T) {
	srv := httptest.NewServer(newMux("test-key", "ws://127.0.0.1:1"))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	for _, origin := range []string{"null", "https://evil.example"} {
		_, _, err := websocket.Dial(ctx, wsURL(srv.URL)+"/voice", &websocket.DialOptions{
			HTTPHeader: http.Header{"Origin": []string{origin}},
		})
		if err == nil {
			t.Errorf("Origin %q was accepted, want rejected", origin)
		}
	}
}

func TestServesIndex(t *testing.T) {
	srv := httptest.NewServer(newMux("test-key", "ws://127.0.0.1:1"))
	defer srv.Close()

	res, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	body, _ := io.ReadAll(res.Body)

	if res.StatusCode != http.StatusOK || !strings.Contains(string(body), "/voice") {
		t.Fatalf("GET / = %d, body contains /voice: %v", res.StatusCode, strings.Contains(string(body), "/voice"))
	}
	if res, _ := http.Get(srv.URL + "/nope"); res.StatusCode != http.StatusNotFound {
		t.Errorf("GET /nope = %d, want 404", res.StatusCode)
	}
}
