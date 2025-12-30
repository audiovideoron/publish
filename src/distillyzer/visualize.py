"""Generate informational images from knowledge base content using Google Gemini."""

import os
from pathlib import Path
import base64

from google import genai
from google.genai import types
from dotenv import load_dotenv

from . import db, embed

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Model for image generation (Nano Banana)
IMAGE_MODEL = "gemini-2.5-flash-image"


def search_context(query: str, num_chunks: int = 3) -> str:
    """Search knowledge base and return relevant context."""
    query_embedding = embed.get_embedding(query)
    chunks = db.search_chunks(query_embedding, limit=num_chunks)

    if not chunks:
        return ""

    context_parts = []
    for chunk in chunks:
        source = chunk["item_title"]
        content = chunk["content"][:500]  # Limit context size
        context_parts.append(f"From '{source}':\n{content}")

    return "\n\n".join(context_parts)


def create_diagram_prompt(concept: str, context: str) -> str:
    """Create a prompt for generating an informational diagram."""
    key_points = context[:400] if context else ""

    return f"""Create a professional infographic diagram that visualizes this concept:

CONCEPT: {concept}

KEY INFORMATION:
{key_points}

STYLE REQUIREMENTS:
- Clean, modern educational infographic
- Use labeled boxes, arrows, and visual hierarchy
- Blue and gray color palette with accent colors
- Minimal text, maximum visual clarity
- Suitable for learning materials

Generate an image that helps someone understand this concept at a glance."""


def generate_image(
    concept: str,
    output_dir: Path | str = Path("images"),
    use_context: bool = True,
    num_chunks: int = 3,
) -> dict:
    """
    Generate an informational image for a concept.

    Args:
        concept: The concept or topic to visualize
        output_dir: Directory to save generated images
        use_context: Whether to search knowledge base for context
        num_chunks: Number of chunks to use for context

    Returns:
        dict with status, path, and any text response
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get context from knowledge base if requested
    context = ""
    if use_context:
        context = search_context(concept, num_chunks)

    # Create prompt
    prompt = create_diagram_prompt(concept, context)

    result = {
        "concept": concept,
        "context_used": bool(context and use_context),
        "text_response": None,
        "image_path": None,
        "status": "no_image_generated",
    }

    try:
        # Generate image using Gemini API
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        # Process response
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                result["text_response"] = part.text
            elif hasattr(part, "inline_data") and part.inline_data:
                # Save image
                filename = concept.lower().replace(" ", "_").replace(":", "")[:30] + ".png"
                image_path = output_dir / filename

                # Decode and save image
                image_data = base64.b64decode(part.inline_data.data)
                with open(image_path, "wb") as f:
                    f.write(image_data)

                result["image_path"] = str(image_path)
                result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["text_response"] = str(e)

    return result


def generate_from_chunk(
    chunk_id: int,
    output_dir: Path | str = Path("images"),
) -> dict:
    """Generate an image based on a specific chunk's content."""
    # Get chunk content directly from DB
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.content, i.title
                FROM chunks c
                JOIN items i ON c.item_id = i.id
                WHERE c.id = %s
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
            if not row:
                return {"status": "error", "text_response": f"Chunk {chunk_id} not found"}

            content, title = row

    # Use chunk content as context
    context = f"From '{title}':\n{content}"

    # Extract a concept from the content (first ~50 chars as title)
    concept = content[:50].strip()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = create_diagram_prompt(concept, context)

    result = {
        "chunk_id": chunk_id,
        "source": title,
        "text_response": None,
        "image_path": None,
        "status": "no_image_generated",
    }

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                result["text_response"] = part.text
            elif hasattr(part, "inline_data") and part.inline_data:
                filename = f"chunk_{chunk_id}.png"
                image_path = output_dir / filename

                image_data = base64.b64decode(part.inline_data.data)
                with open(image_path, "wb") as f:
                    f.write(image_data)

                result["image_path"] = str(image_path)
                result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["text_response"] = str(e)

    return result
