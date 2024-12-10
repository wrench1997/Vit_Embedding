
import torch
from PIL import Image
from embed.net_emded import load_embedding_model,prepare_transform,get_embedding



# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 64, 36)
    embed_dim = 2048
    hidden_dim = 64

    encoder_path = 'checkpoint/encoder_epoch_latset.pth'
    projection_head_path = 'checkpoint/proj_head_epoch_latset.pth'

    encoder, projection_head = load_embedding_model(encoder_path, projection_head_path, input_shape, embed_dim, hidden_dim, device)

    transform = prepare_transform()
    image_path = './data/output_dir/val/initial_segment_1.png'  # Replace with your image path
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    embedding = get_embedding(encoder, projection_head, image, device)
    print(f"Embedding for {image_path}: {embedding}")
    print(f"Maximum value in embedding: {torch.max(torch.tensor(embedding))}")
    print(f"Minimum value in embedding: {torch.min(torch.tensor(embedding))}")

