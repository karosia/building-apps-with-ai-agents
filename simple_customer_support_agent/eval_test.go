//go:build eval

// Minimal evaluation against the real OpenAI API.
// Excluded from `go test ./...` by the build tag; run it explicitly:
//
//	go test -tags eval -v ./simple_customer_support_agent
package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/langchaingo/llms"
)

func TestMinimalEval(t *testing.T) {
	loadEnv()
	llm, err := newLLM()
	if err != nil {
		t.Skipf("skipping eval: %v", err)
	}

	app, err := constructGraph(llm)
	if err != nil {
		t.Fatalf("compile graph: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	result, err := app.Invoke(ctx, AgentState{
		Order: map[string]string{"order_id": "B73973"},
		Messages: []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman,
				"I found a cheaper option elsewhere. Please cancel order #B73973."),
		},
	})
	if err != nil {
		t.Fatalf("run graph: %v", err)
	}

	for _, m := range result.Messages {
		t.Logf("%s: %s", m.Role, contentString(m))
	}

	// Check 1: the cancel_order tool was called (an AI tool call or a tool message exists)
	if !hasToolCall(result.Messages, "cancel_order") {
		t.Fatal("the cancel_order tool was not called")
	}

	// Check 2: the final AI reply confirms the cancellation
	reply := lastAIText(result.Messages)
	if !strings.Contains(strings.ToLower(reply), "cancel") {
		t.Fatalf("confirmation message is missing; final reply: %q", reply)
	}

	t.Log("✅ The agent passed the minimum evaluation criteria.")
}

// hasToolCall reports whether any message contains a call to the named tool or its result.
func hasToolCall(msgs []llms.MessageContent, name string) bool {
	for _, m := range msgs {
		for _, p := range m.Parts {
			switch v := p.(type) {
			case llms.ToolCall:
				if v.FunctionCall != nil && v.FunctionCall.Name == name {
					return true
				}
			case llms.ToolCallResponse:
				if v.Name == name {
					return true
				}
			}
		}
	}
	return false
}

// lastAIText returns the text of the last AI message, or "" if there is none.
func lastAIText(msgs []llms.MessageContent) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llms.ChatMessageTypeAI {
			continue
		}
		var sb strings.Builder
		for _, p := range msgs[i].Parts {
			if tc, ok := p.(llms.TextContent); ok {
				sb.WriteString(tc.Text)
			}
		}
		if sb.Len() > 0 {
			return sb.String()
		}
	}
	return ""
}
