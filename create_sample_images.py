"""
Script to create sample placeholder images for the RAG-based AI Tutor
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image(filename, title, width=400, height=300):
    """Create a simple placeholder image with title text"""
    # Create image with light background
    img = Image.new('RGB', (width, height), color='#f0f8ff')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        title_font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_x = (width - title_width) // 2
    title_y = height // 3
    
    draw.text((title_x, title_y), title, fill='#333', font=title_font)
    
    # Draw a simple border
    draw.rectangle([10, 10, width-10, height-10], outline='#4169e1', width=3)
    
    # Add some decorative elements
    draw.ellipse([width//4, height//2, 3*width//4, 4*height//5], outline='#4169e1', width=2)
    
    # Save image
    img.save(f'static/{filename}')
    print(f"Created {filename}")

# Create sample images based on the metadata
images_to_create = [
    ("bell.png", "Bell Vibration"),
    ("sound_waves.png", "Sound Wave Propagation"),
    ("frequency_amplitude.png", "Frequency & Amplitude"),
    ("human_ear.png", "Human Ear Structure"),
    ("echo_reflection.png", "Echo & Sound Reflection"),
    ("musical_instruments.png", "Musical Instruments"),
    ("ultrasound_infrasound.png", "Ultrasound & Infrasound"),
    ("doppler_effect.png", "Doppler Effect")
]

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    for filename, title in images_to_create:
        create_sample_image(filename, title)
    
    print(f"\nCreated {len(images_to_create)} sample images in the static directory!")
    print("You can replace these with actual educational diagrams for better results.")