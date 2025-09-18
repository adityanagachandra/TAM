# Token Activation Map (TAM) Visual Explanation Task

## Task Description
Your goal is to use Token Activation Map (TAM) to generate visual explanations for multimodal Large Language Model (LLM) outputs. TAM reveals which parts of an image contribute to each word the model generates, providing clear visualizations of the model's attention patterns.

## Input
- 10 COCO images (JPG format) in the `input/image/` directory
- Segmentation labels for each image in `input/seg_label/`
- COCO annotations in `input/annotations/`

## Requirements
1. **Load a Multimodal LLM**: Use Qwen2-VL-2B-Instruct or a similar multimodal model
2. **Generate Image Descriptions**: For each input image, generate a natural language description
3. **Apply TAM Analysis**: Use the Token Activation Map technique to create visual explanations showing:
   - Which image regions correspond to each generated token
   - The strength of activation for each token-region pair
4. **Evaluate Results**: Calculate IoU and F1 scores comparing TAM activations to ground truth segmentations

## Output Requirements
Save your results in a file named `tam_results.json` with the following structure:

```json
{
  "results": [
    {
      "image": "000000121031.jpg",
      "description": "A cat sitting on a couch near a window",
      "token_activations": [
        {
          "token": "cat",
          "activation_map": "visualizations/000000121031_cat.jpg",
          "peak_coordinates": [120, 80],
          "activation_strength": 0.92
        },
        {
          "token": "couch", 
          "activation_map": "visualizations/000000121031_couch.jpg",
          "peak_coordinates": [200, 150],
          "activation_strength": 0.85
        }
      ],
      "metrics": {
        "iou": 0.78,
        "f1_score": 0.82,
        "noun_recall": 0.85,
        "function_word_accuracy": 0.90
      }
    }
  ],
  "overall_metrics": {
    "average_iou": 0.75,
    "average_f1": 0.80,
    "average_noun_recall": 0.83,
    "average_function_accuracy": 0.88
  }
}
```

## Key Implementation Steps
1. **Model Setup**: Load the multimodal LLM with hidden state output enabled
2. **TAM Configuration**: Set up special token IDs and vision shape parameters
3. **Visualization**: Generate activation maps for each meaningful token
4. **Evaluation**: Compare TAM outputs against ground truth segmentations

## Evaluation Criteria
- Minimum IoU score: 0.5
- Minimum F1 score: 0.6
- All 10 images must be processed
- Visualizations must be saved for inspection

## Notes
- Use the provided segmentation labels for quantitative evaluation
- The TAM method should produce clear, interpretable visualizations

## References
- Use the provided `tam.py` module for core TAM functionality
