from pydantic import BaseModel, Field


class GenerationPayload(BaseModel):
    """
    Payload for generating images or video.
    """

    text_prompt: str = Field(
        # pylint: disable=line-too-long
        "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
        title="A description of what you'd like the machine to generate. Think of it like writing the caption below your image on a website.",
    )


class ImageGenerationResult(BaseModel):
    """
    Image Generation Result about user Request. Result is encoded by Base64 format.
    """

    text_promt: str
    result: str
