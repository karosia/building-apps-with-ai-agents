// realtime_voice_agent bridges a browser WebSocket to the OpenAI Realtime API.
//
//  1. The browser opens http://host:5050 (index.html, served by this server) and connects to ws://host:5050/voice
//  2. The server opens a WebSocket to the OpenAI Realtime API and initializes the session
//  3. fromClient: browser → OpenAI, streams PCM16 microphone audio (input_audio_buffer.append)
//  4. toClient:   OpenAI → browser, relays assistant audio deltas (response.output_audio.delta)
//  5. Interruption: when the user starts speaking, the assistant reply is cut off (conversation.item.truncate)
package main

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/joho/godotenv"
)

const (
	realtimeURL   = "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini"
	voice         = "coral" // voice names are lowercase
	pcmSampleRate = 24000   // PCM16 mono at 24 kHz, the rate the Realtime API expects
	port          = 5050

	// Max size of a single WebSocket message we accept (OpenAI caps audio chunks at 15 MB).
	readLimit = 16 << 20

	instructions = "You are a concise AI agent. Respond in Korean. " +
		"Be as polite as possible. Keep your answers as short as possible."
)

// clientMessage is the JSON exchanged with the browser in both directions: {"audio": "<base64 PCM16>"}.
type clientMessage struct {
	Audio string `json:"audio"`
}

// serverEvent holds the fields we use from OpenAI Realtime server events.
type serverEvent struct {
	Type   string `json:"type"`
	Delta  string `json:"delta"`   // response.output_audio.delta: base64 audio chunk
	ItemID string `json:"item_id"` // id of the assistant item that produced the audio
	Error  *struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// sessionUpdate builds the session configuration (GA Realtime API format).
func sessionUpdate() map[string]any {
	pcm := map[string]any{"type": "audio/pcm", "rate": pcmSampleRate}
	return map[string]any{
		"type": "session.update",
		"session": map[string]any{
			"type":              "realtime",
			"output_modalities": []string{"audio"},
			"instructions":      instructions,
			"audio": map[string]any{
				"input": map[string]any{
					"format":         pcm,
					"turn_detection": map[string]any{"type": "server_vad"},
				},
				"output": map[string]any{
					"format": pcm,
					"voice":  voice,
				},
			},
		},
	}
}

// voiceBridge handles one browser connection for its whole lifetime.
func voiceBridge(apiKey, upstreamURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// The default origin check only accepts pages served from this same host,
		// which is why index.html is served by this server instead of opened from disk.
		client, err := websocket.Accept(w, r, nil)
		if err != nil {
			log.Printf("accept browser connection: %v", err)
			return
		}
		defer client.CloseNow()
		client.SetReadLimit(readLimit)

		// Cancelling ctx stops both relay loops and closes both connections.
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()

		// Open the upstream connection to the OpenAI Realtime API
		upstream, _, err := websocket.Dial(ctx, upstreamURL, &websocket.DialOptions{
			HTTPHeader: http.Header{"Authorization": []string{"Bearer " + apiKey}},
		})
		if err != nil {
			log.Printf("connect to OpenAI Realtime API: %v", err)
			client.Close(websocket.StatusInternalError, "upstream unavailable")
			return
		}
		defer upstream.CloseNow()
		upstream.SetReadLimit(readLimit)

		// Initialize the Realtime session
		if err := wsjson.Write(ctx, upstream, sessionUpdate()); err != nil {
			log.Printf("send session.update: %v", err)
			return
		}

		// Run both directions concurrently (asyncio.gather equivalent).
		// Unlike gather, the session ends as soon as either side finishes.
		errc := make(chan error, 2)
		go func() { errc <- fromClient(ctx, client, upstream) }()
		go func() { errc <- toClient(ctx, upstream, client) }()

		if err := <-errc; err != nil && !isClosed(err) {
			log.Printf("voice session ended: %v", err)
		}
		cancel()
		<-errc // wait for the other loop to exit before the deferred closes run
	}
}

// fromClient relays microphone PCM chunks from the browser to OpenAI.
func fromClient(ctx context.Context, client, upstream *websocket.Conn) error {
	for {
		var msg clientMessage
		if err := wsjson.Read(ctx, client, &msg); err != nil {
			return err
		}
		if msg.Audio == "" {
			continue
		}
		// The audio is already base64, so forward it as-is instead of decoding and re-encoding.
		if err := wsjson.Write(ctx, upstream, map[string]string{
			"type":  "input_audio_buffer.append",
			"audio": msg.Audio,
		}); err != nil {
			return err
		}
	}
}

// toClient relays assistant audio to the browser and handles interruptions.
func toClient(ctx context.Context, upstream, client *websocket.Conn) error {
	// Only this goroutine touches lastAssistantItem, so no lock is needed.
	var lastAssistantItem string

	for {
		var ev serverEvent
		if err := wsjson.Read(ctx, upstream, &ev); err != nil {
			return err
		}

		switch ev.Type {
		// The assistant is speaking
		case "response.output_audio.delta":
			if err := wsjson.Write(ctx, client, clientMessage{Audio: ev.Delta}); err != nil {
				return err
			}
			lastAssistantItem = ev.ItemID

		// The user started speaking → stop the assistant's reply
		case "input_audio_buffer.speech_started":
			if lastAssistantItem == "" {
				continue
			}
			// coder/websocket allows concurrent writes, so this is safe alongside fromClient's appends.
			if err := wsjson.Write(ctx, upstream, map[string]any{
				"type":          "conversation.item.truncate",
				"item_id":       lastAssistantItem,
				"content_index": 0,
				"audio_end_ms":  0, // cut off immediately
			}); err != nil {
				return err
			}
			lastAssistantItem = ""

		// Surface API errors (bad session config, invalid voice, etc.) instead of dropping them
		case "error":
			if ev.Error != nil {
				log.Printf("OpenAI error (%s): %s", ev.Error.Code, ev.Error.Message)
			}
		}
	}
}

// isClosed reports whether err is just one side hanging up or the session being cancelled.
func isClosed(err error) bool {
	if errors.Is(err, context.Canceled) {
		return true
	}
	switch websocket.CloseStatus(err) {
	case websocket.StatusNormalClosure, websocket.StatusGoingAway, websocket.StatusNoStatusRcvd:
		return true
	}
	return false
}

// loadEnv loads .env from the current directory or the repo root, whichever exists.
func loadEnv() {
	for _, f := range []string{".env", "../.env"} {
		if godotenv.Load(f) == nil {
			return
		}
	}
}

func main() {
	loadEnv()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set; set it as an environment variable or in a .env file")
	}

	line := strings.Repeat("=", 60)
	fmt.Printf("\n%s\n\nRealtime voice agent server is starting...\n", line)
	fmt.Printf("\nOpen http://localhost:%d in your browser.\n\n%s\n\n", port, line)

	log.Fatal(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", port), newMux(apiKey, realtimeURL)))
}

// newMux wires the routes: the browser client at / and the voice bridge at /voice.
func newMux(apiKey, upstreamURL string) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", serveIndex) // {$} matches "/" exactly, not every path
	mux.HandleFunc("GET /voice", voiceBridge(apiKey, upstreamURL))
	return mux
}

// indexHTML is the browser client, compiled into the binary.
//
//go:embed index.html
var indexHTML []byte

func serveIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}
