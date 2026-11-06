package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/smallnest/langgraphgo/graph"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

// AgentState defines the graph state.
// Python's Annotated[Sequence[BaseMessage], operator.add] is expressed via the schema below (appendMessages).
type AgentState struct {
	Order    map[string]string
	Messages []llms.MessageContent
}

// appendMessages appends the messages returned by a node to the existing messages (equivalent to operator.add).
// Order is overwritten only when a new value is provided.
func appendMessages(current, update AgentState) (AgentState, error) {
	if update.Order != nil {
		current.Order = update.Order
	}
	current.Messages = append(current.Messages, update.Messages...)
	return current, nil
}

// -- 1) Define the order cancellation tool
// Implements langchaingo's tools.Tool interface (Name/Description/Call).
type CancelOrderTool struct{}

var _ tools.Tool = CancelOrderTool{}

func (CancelOrderTool) Name() string        { return "cancel_order" }
func (CancelOrderTool) Description() string { return "Cancels an order that has not been shipped yet." }

// Call receives the JSON arguments from the LLM ({"order_id": "..."}) and executes the tool.
func (CancelOrderTool) Call(ctx context.Context, input string) (string, error) {
	var args struct {
		OrderID string `json:"order_id"`
	}
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		return "", fmt.Errorf("parse cancel_order arguments: %w", err)
	}
	// (Call the real backend API here)
	return fmt.Sprintf("Order %s has been cancelled.", args.OrderID), nil
}

// Schema returns the JSON schema of the tool's arguments (what Python's @tool generated automatically).
// It also satisfies langgraphgo's prebuilt.ToolWithSchema, so the same tool works with prebuilt agents.
func (CancelOrderTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"order_id": map[string]any{
				"type":        "string",
				"description": "The order ID to cancel",
			},
		},
		"required": []string{"order_id"},
	}
}

// Definition builds the llms.Tool bound to the LLM from Schema().
func (t CancelOrderTool) Definition() llms.Tool {
	return llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Schema(),
		},
	}
}

var cancelOrder = CancelOrderTool{}

// -- 2) Define the agent logic: call the LLM, run the tool, call the LLM again
func callModel(llm llms.Model) func(ctx context.Context, state AgentState) (AgentState, error) {
	return func(ctx context.Context, state AgentState) (AgentState, error) {
		orderID := "UNKNOWN"
		if id, ok := state.Order["order_id"]; ok && id != "" {
			orderID = id
		}

		// The system prompt tells the model exactly what to do
		prompt := fmt.Sprintf(`You are an e-commerce support agent.
Order ID: %s
If the customer asks to cancel, call cancel_order(order_id)
and send a short confirmation message.
Otherwise, respond normally.`, orderID)

		full := append([]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, prompt),
		}, state.Messages...)

		// First LLM pass: decide whether to call a tool (bind_tools equivalent: llms.WithTools)
		first, err := llm.GenerateContent(ctx, full,
			llms.WithTemperature(0),
			llms.WithTools([]llms.Tool{cancelOrder.Definition()}),
		)
		if err != nil {
			return AgentState{}, fmt.Errorf("first LLM call: %w", err)
		}
		if len(first.Choices) == 0 {
			return AgentState{}, errors.New("first LLM call returned no choices")
		}
		choice := first.Choices[0]

		var out []llms.MessageContent

		if len(choice.ToolCalls) == 0 {
			out = append(out, llms.TextParts(llms.ChatMessageTypeAI, choice.Content))
			return AgentState{Messages: out}, nil
		}

		// Run the cancel_order tool (like the Python version, only the first tool call is handled)
		tc := choice.ToolCalls[0]
		aiMsg := llms.MessageContent{Role: llms.ChatMessageTypeAI}
		if choice.Content != "" {
			aiMsg.Parts = append(aiMsg.Parts, llms.TextContent{Text: choice.Content})
		}
		aiMsg.Parts = append(aiMsg.Parts, tc)
		out = append(out, aiMsg)

		result, err := cancelOrder.Call(ctx, tc.FunctionCall.Arguments)
		if err != nil {
			return AgentState{}, err
		}
		out = append(out, llms.MessageContent{
			Role: llms.ChatMessageTypeTool,
			Parts: []llms.ContentPart{llms.ToolCallResponse{
				ToolCallID: tc.ID,
				Name:       tc.FunctionCall.Name,
				Content:    result,
			}},
		})

		// Second LLM pass: generate the final confirmation text (without tools bound)
		second, err := llm.GenerateContent(ctx, append(full, out...), llms.WithTemperature(0))
		if err != nil {
			return AgentState{}, fmt.Errorf("second LLM call: %w", err)
		}
		if len(second.Choices) == 0 {
			return AgentState{}, errors.New("second LLM call returned no choices")
		}
		out = append(out, llms.TextParts(llms.ChatMessageTypeAI, second.Choices[0].Content))

		return AgentState{Messages: out}, nil
	}
}

// -- 3) Wire up the agent with a StateGraph
func constructGraph(llm llms.Model) (*graph.StateRunnable[AgentState], error) {
	g := graph.NewStateGraph[AgentState]()
	g.SetSchema(graph.NewStructSchema(AgentState{}, appendMessages))
	g.AddNode("assistant", "Call LLM + run tool", callModel(llm))
	g.SetEntryPoint("assistant")
	g.AddEdge("assistant", graph.END) // langgraphgo errors on nodes with no outgoing edge, so add it explicitly
	return g.Compile()
}

// loadEnv loads .env from the current directory or the repo root, whichever exists.
// godotenv.Load(a, b) stops at the first missing file, so try each one separately.
func loadEnv() {
	for _, f := range []string{".env", "../.env"} {
		if godotenv.Load(f) == nil {
			return
		}
	}
}

// newLLM creates the OpenAI chat model (init_chat_model equivalent).
func newLLM() (llms.Model, error) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		return nil, errors.New("OPENAI_API_KEY is not set; set it as an environment variable or in a .env file")
	}
	return openai.New(openai.WithModel("gpt-5-mini"))
}

func main() {
	loadEnv()

	// Created once and injected into the graph
	llm, err := newLLM()
	if err != nil {
		log.Fatal(err)
	}

	app, err := constructGraph(llm)
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}

	ctx := context.Background()
	result, err := app.Invoke(ctx, AgentState{
		Order: map[string]string{"order_id": "B73973"},
		Messages: []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Please cancel order #B73973."),
		},
	})
	if err != nil {
		log.Fatalf("run graph: %v", err)
	}

	for _, msg := range result.Messages {
		fmt.Printf("%s: %s\n", msg.Role, contentString(msg))
	}
}

// contentString converts a message's parts into a human-readable string.
func contentString(msg llms.MessageContent) string {
	var s string
	for _, p := range msg.Parts {
		switch v := p.(type) {
		case llms.TextContent:
			s += v.Text
		case llms.ToolCall:
			s += fmt.Sprintf("[tool_call %s(%s)]", v.FunctionCall.Name, v.FunctionCall.Arguments)
		case llms.ToolCallResponse:
			s += v.Content
		}
	}
	return s
}
