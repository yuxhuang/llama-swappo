package proxy

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/mostlygeek/llama-swap/proxy/config"
)

func TestParseHFConfig(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hf-model-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	configJSON := `{
		"architectures": ["Qwen2ForCausalLM"],
		"model_type": "qwen2",
		"max_position_embeddings": 32768,
		"quantization_config": {
			"quant_method": "awq",
			"bits": 4
		},
		"transformers_version": "4.45.0"
	}`

	err = os.WriteFile(filepath.Join(tmpDir, "config.json"), []byte(configJSON), 0644)
	if err != nil {
		t.Fatal(err)
	}

	hf, err := ParseHFConfig(tmpDir)
	assert.NoError(t, err)
	assert.Equal(t, "qwen2", hf.ModelType)
	assert.Equal(t, 32768, hf.MaxPositionEmbeddings)
	assert.Equal(t, "awq", hf.Quantization["quant_method"])
}

func TestLlamaServerParser_HFDirectory(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hf-model-dir-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	parser := NewLlamaServerParser()
	cmd := fmt.Sprintf("vllm --model \"%s\" --max-model-len 8192", tmpDir)
	
	args := parser.Parse(cmd, "test-model")
	
	assert.True(t, args.IsDirectory, "Expected IsDirectory to be true for " + tmpDir)
	assert.Equal(t, tmpDir, args.FullModelPath)
	assert.Equal(t, 8192, args.ContextLength)
	assert.Equal(t, "safetensors", args.Format)
}

func TestProxyManager_GetModelDetails_HF(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "hf-model-details-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	configJSON := `{
		"architectures": ["LlamaForCausalLM"],
		"max_position_embeddings": 4096,
		"torch_dtype": "float16"
	}`
	os.WriteFile(filepath.Join(tmpDir, "config.json"), []byte(configJSON), 0644)

	pm := &ProxyManager{
		config: config.Config{
			Models: make(map[string]config.ModelConfig),
		},
		modelInfoCache: make(map[string]struct {
			Details      OllamaModelDetails
			Capabilities []string
		}),
	}

	modelCfg := config.ModelConfig{
		Cmd: fmt.Sprintf("vllm --model \"%s\"", tmpDir),
	}

	details, _ := pm.getModelDetails(modelCfg, "llama-hf")

	assert.Equal(t, "LlamaForCausalLM", details.Family)
	assert.Equal(t, 4096, details.ContextLength)
	assert.Equal(t, "FLOAT16", details.QuantizationLevel)
	assert.Equal(t, "safetensors", details.Format)
}
