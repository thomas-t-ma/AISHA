from pydantic import BaseModel


class ProviderCapabilities(BaseModel):
    streaming_text: bool = True
    tools: bool = False
    vision: bool = False
    structured_output: bool = False
    remote: bool = False
