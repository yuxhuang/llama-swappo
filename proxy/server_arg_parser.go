package proxy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

// ServerArgs holds information parsed or inferred from server command line arguments.
type ServerArgs struct {
	Architecture      string
	ContextLength     int
	Capabilities      []string
	Family            string
	ParameterSize     string
	QuantizationLevel string
	CmdAlias          string
	CmdModelPath      string // Base name of the model file from --model arg
	FullModelPath     string // Full path to the model file from --model arg
	Format            string // gguf, safetensors, transformers, etc.
	IsDirectory       bool   // True if FullModelPath points to a directory (HF model)
}

// ServerArgParser defines an interface for parsing server command line arguments.
type ServerArgParser interface {
	Parse(cmdStr string, modelID string) ServerArgs
}

// LlamaServerParser implements ServerArgParser for llama-server.
type LlamaServerParser struct{}

var (
	architecturePatterns = map[string]*regexp.Regexp{
		"command-r": regexp.MustCompile(`(?i)command-r`),
		"gemma2":    regexp.MustCompile(`(?i)gemma2`),
		"gemma3":    regexp.MustCompile(`(?i)gemma3`),
		"gemma":     regexp.MustCompile(`(?i)gemma`),
		"llama4":    regexp.MustCompile(`(?i)llama-?4`),
		"llama3":    regexp.MustCompile(`(?i)llama-?3`),
		"llama":     regexp.MustCompile(`(?i)llama`),
		"mistral3":  regexp.MustCompile(`(?i)mistral-?3`),
		"mistral":   regexp.MustCompile(`(?i)mistral`),
		"phi3":      regexp.MustCompile(`(?i)phi-?3`),
		"phi":       regexp.MustCompile(`(?i)phi`),
		"qwen2.5vl": regexp.MustCompile(`(?i)qwen-?2\.5-?vl`),
		"qwen3":     regexp.MustCompile(`(?i)qwen-?3`),
		"qwen2":     regexp.MustCompile(`(?i)qwen-?2`),
		"qwen":      regexp.MustCompile(`(?i)qwen`),
		"bert":      regexp.MustCompile(`(?i)bert`),
		"clip":      regexp.MustCompile(`(?i)clip`),
	}
	orderedArchKeys = []string{
		"command-r", "gemma3", "gemma2", "gemma", "llama4", "llama3", "llama",
		"mistral3", "mistral", "phi3", "phi", "qwen2.5vl", "qwen3", "qwen2", "qwen",
		"bert", "clip",
	}

	parameterSizePattern = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?(?:x\d+)?)[BMGT]?B`)
	quantizationPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)IQ[1-4]_(XXS|XS|S|M|NL)`),
		regexp.MustCompile(`(?i)Q[2-8]_(0|1|[KSLM]+(?:_[KSLM]+)?)`),
		regexp.MustCompile(`(?i)BPW\d+`),
		regexp.MustCompile(`(?i)GGML_TYPE_Q[2-8]_\d`),
		regexp.MustCompile(`(?i)F(?:P)?(16|32)`),
		regexp.MustCompile(`(?i)BF16`),
	}
)

func inferPattern(name string, patterns map[string]*regexp.Regexp, orderedKeys []string) string {
	nameLower := strings.ToLower(name)
	for _, key := range orderedKeys {
		pattern, ok := patterns[key]
		if !ok || pattern == nil {
			continue
		}
		if pattern.MatchString(nameLower) {
			return key
		}
	}
	return "unknown"
}

func inferQuantizationLevelFromName(name string) string {
	for _, pattern := range quantizationPatterns {
		match := pattern.FindString(name)
		if match != "" {
			return strings.ToUpper(match)
		}
	}
	return "unknown"
}

func inferParameterSizeFromName(name string) string {
	match := parameterSizePattern.FindStringSubmatch(name)
	if len(match) > 0 {
		return strings.ToUpper(match[0])
	}
	return "unknown"
}

func inferFamilyFromName(nameForInference string, currentArch string) string {
	if currentArch != "unknown" && currentArch != "" {
		re := regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*)`)
		match := re.FindStringSubmatch(currentArch)
		if len(match) > 1 {
			potentialFamily := strings.ToLower(match[1])
			knownFamilies := []string{"llama", "qwen", "phi", "mistral", "gemma", "command-r", "bert", "clip"}
			for _, kf := range knownFamilies {
				if potentialFamily == kf {
					return kf
				}
			}
			for _, kf := range knownFamilies {
				if strings.ToLower(currentArch) == kf {
					return kf
				}
			}
		}
	}
	orderedFamilyCheckKeys := []string{"command-r", "gemma", "llama", "mistral", "phi", "qwen", "bert", "clip"}
	familyPatterns := make(map[string]*regexp.Regexp)
	for _, key := range orderedFamilyCheckKeys {
		if p, ok := architecturePatterns[key]; ok {
			familyPatterns[key] = p
		}
	}
	return inferPattern(nameForInference, familyPatterns, orderedFamilyCheckKeys)
}

// NewLlamaServerParser creates a new parser for llama-server arguments.
func NewLlamaServerParser() *LlamaServerParser {
	return &LlamaServerParser{}
}

// Parse extracts relevant information from llama-server command string and modelID.
func (p *LlamaServerParser) Parse(cmdStr string, modelID string) ServerArgs {
	parsed := ServerArgs{
		Capabilities: []string{"completion"}, // Default
	}

	args, err := config.SanitizeCommand(cmdStr)
	if err != nil {
		// If sanitization fails, proceed with inference based on modelID only
		parsed.Architecture = inferPattern(modelID, architecturePatterns, orderedArchKeys)
		parsed.Family = inferFamilyFromName(modelID, parsed.Architecture)
		parsed.ParameterSize = inferParameterSizeFromName(modelID)
		parsed.QuantizationLevel = inferQuantizationLevelFromName(modelID)
		return parsed
	}

	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch arg {
		case "-c", "--ctx-size":
			if i+1 < len(args) {
				if valInt, err := strconv.Atoi(args[i+1]); err == nil {
					parsed.ContextLength = valInt
				}
				i++
			}
		case "-a", "--alias":
			if i+1 < len(args) {
				parsed.CmdAlias = args[i+1]
				i++
			}
		case "-m", "--model":
			if i+1 < len(args) {
				parsed.FullModelPath = args[i+1]
				parsed.CmdModelPath = filepath.Base(args[i+1])
				// check if it is a directory
				if info, err := os.Stat(parsed.FullModelPath); err == nil && info.IsDir() {
					parsed.IsDirectory = true
					parsed.Format = "safetensors" // assume HF directory initially
				}
				if parsed.Format == "" && strings.HasSuffix(parsed.FullModelPath, ".gguf") {
					parsed.Format = "gguf"
				}
				i++
			}
		case "--max-model-len": // vllm specific
			if i+1 < len(args) {
				if valInt, err := strconv.Atoi(args[i+1]); err == nil {
					parsed.ContextLength = valInt
				}
				i++
			}
		case "--served-model-name": // vllm specific
			if i+1 < len(args) {
				parsed.CmdAlias = args[i+1]
				i++
			}
		case "--trust-remote-code": // vllm/hf specific
			foundTools := false
			for _, cap := range parsed.Capabilities {
				if cap == "tools" {
					foundTools = true
					break
				}
			}
			if !foundTools {
				parsed.Capabilities = append(parsed.Capabilities, "tools")
			}
		case "--jinja":
			foundTools := false
			for _, cap := range parsed.Capabilities {
				if cap == "tools" {
					foundTools = true
					break
				}
			}
			if !foundTools {
				parsed.Capabilities = append(parsed.Capabilities, "tools")
			}
		case "--mmproj":
			foundVision := false
			for _, cap := range parsed.Capabilities {
				if cap == "vision" {
					foundVision = true
					break
				}
			}
			if !foundVision {
				parsed.Capabilities = append(parsed.Capabilities, "vision")
			}
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				i++
			}
		}
	}

	parsed.Architecture = inferPattern(modelID, architecturePatterns, orderedArchKeys)
	if parsed.Architecture == "unknown" {
		parsed.Architecture = inferPattern(parsed.CmdAlias, architecturePatterns, orderedArchKeys)
	}
	if parsed.Architecture == "unknown" {
		parsed.Architecture = inferPattern(parsed.CmdModelPath, architecturePatterns, orderedArchKeys)
	}

	parsed.Family = inferFamilyFromName(modelID, parsed.Architecture)
	if parsed.Family == "unknown" {
		parsed.Family = inferFamilyFromName(parsed.CmdAlias, parsed.Architecture)
	}
	if parsed.Family == "unknown" {
		parsed.Family = inferFamilyFromName(parsed.CmdModelPath, parsed.Architecture)
	}

	parsed.ParameterSize = inferParameterSizeFromName(modelID)
	if parsed.ParameterSize == "unknown" {
		parsed.ParameterSize = inferParameterSizeFromName(parsed.CmdAlias)
	}
	if parsed.ParameterSize == "unknown" {
		parsed.ParameterSize = inferParameterSizeFromName(parsed.CmdModelPath)
	}

	parsed.QuantizationLevel = inferQuantizationLevelFromName(modelID)
	if parsed.QuantizationLevel == "unknown" {
		parsed.QuantizationLevel = inferQuantizationLevelFromName(parsed.CmdAlias)
	}
	if parsed.QuantizationLevel == "unknown" {
		parsed.QuantizationLevel = inferQuantizationLevelFromName(parsed.CmdModelPath)
	}

	return parsed
}

// HFConfig represents relevant parts of config.json
type HFConfig struct {
	Architecture          []string       `json:"architectures"`
	ModelType             string         `json:"model_type"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	ContextLength         int            `json:"context_length"`          // Some models use this
	MaxSeqLen             int            `json:"max_sequence_length"`     // Some models use this
	ModelMaxLen           int            `json:"model_max_length"`        // Some models use this
	NumHiddenLayers       int            `json:"num_hidden_layers"`       // for param inference
	HiddenSize            int            `json:"hidden_size"`             // for param inference
	IntermediateSize      int            `json:"intermediate_size"`       // for param inference
	NumAttentionHeads     int            `json:"num_attention_heads"`     // for param inference
	NumKeyValueHeads      int            `json:"num_key_value_heads"`     // for param inference
	VocabSize             int            `json:"vocab_size"`              // for param inference
	Quantization          map[string]any `json:"quantization_config"`     // to detect AWQ/GPTQ/bitsandbytes
	TorchDataType         string         `json:"torch_dtype"`             // for precision
	TransformerVersion    string         `json:"transformers_version"`    // to check if it is HF
}

// ParseHFConfig reads and parses config.json from a directory
func ParseHFConfig(dirPath string) (HFConfig, error) {
	configPath := filepath.Join(dirPath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return HFConfig{}, err
	}

	var hf HFConfig
	if err := json.Unmarshal(data, &hf); err != nil {
		return HFConfig{}, err
	}

	return hf, nil
}
