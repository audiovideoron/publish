"""Generate informational images from knowledge base content using Google Gemini."""

import os
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

from . import db, embed

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Model for image generation - configurable via environment variable
IMAGE_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-preview-image-generation")


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
    if context:
        return f"""Create a clear, professional diagram for: {concept}

Use this context to inform the diagram:
{context[:400]}

Style: clean, professional, easy to read. Use appropriate colors and layout for the subject matter."""
    else:
        return f"""Create a clear, professional diagram for: {concept}

Style: clean, professional, easy to read. Use appropriate colors and layout for the subject matter."""


def create_artistic_prompt(concept: str, context: str) -> str:
    """Create a prompt for generating artistic/creative imagery."""
    if context:
        return f"""{concept}

Draw inspiration from this context:
{context[:400]}

Create an evocative, artistic image. Focus on mood, atmosphere, and emotional resonance rather than literal representation."""
    else:
        return f"""{concept}

Create an evocative, artistic image. Focus on mood, atmosphere, and emotional resonance rather than literal representation."""


def generate_image(
    concept: str,
    output_dir: Path | str = Path("images"),
    use_context: bool = True,
    num_chunks: int = 3,
    artistic: bool = False,
) -> dict:
    """
    Generate an image for a concept.

    Args:
        concept: The concept or topic to visualize
        output_dir: Directory to save generated images
        use_context: Whether to search knowledge base for context
        num_chunks: Number of chunks to use for context
        artistic: If True, generate artistic/creative imagery instead of diagrams

    Returns:
        dict with status, path, and any text response
    """
    output_dir = Path(output_dir)
    # Directory created lazily only when image is saved

    # Get context from knowledge base if requested
    context = ""
    if use_context:
        context = search_context(concept, num_chunks)

    # Create prompt based on mode
    if artistic:
        prompt = create_artistic_prompt(concept, context)
    else:
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
                # Save image - create directory only when we have an image to save
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = concept.lower().replace(" ", "_").replace(":", "")[:30] + ".png"
                image_path = output_dir / filename

                # Save image (data is already bytes)
                with open(image_path, "wb") as f:
                    f.write(part.inline_data.data)

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
    # Directory created lazily only when image is saved

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
                # Create directory only when we have an image to save
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = f"chunk_{chunk_id}.png"
                image_path = output_dir / filename

                with open(image_path, "wb") as f:
                    f.write(part.inline_data.data)

                result["image_path"] = str(image_path)
                result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["text_response"] = str(e)

    return result
