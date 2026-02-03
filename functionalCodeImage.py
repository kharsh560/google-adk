import os
from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Part
from google.adk.models import LlmRequest
from google.genai.types import Part
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
import asyncio
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from google.adk.artifacts import InMemoryArtifactService
from google.genai.types import Part, Blob
import uuid

artifact_service = InMemoryArtifactService()

from dotenv import load_dotenv
load_dotenv()

APP_NAME = "scorecard_reader_app"
USER_ID = "user_1"
SESSION_ID = "session_001" 
ARTIFACT_ID=None

MODEL_GEMINI_2_5_FLASH = "gemini-2.5-flash"

def load_image_as_binary(filename: str = "imgs.jpeg") -> tuple[bytes, str]:
    """Load an image file from the same directory and return its binary data and MIME type.
    
    Args:
        filename (str): Name of the image file to load (defaults to "imgs.jpeg")
    
    Returns:
        tuple[bytes, str]: Binary data of the image and its MIME type
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, filename)
    
    # Read the image file as binary
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Determine MIME type based on file extension
    file_ext = filename.lower().split('.')[-1]
    if file_ext in ['jpg', 'jpeg']:
        mime_type = 'image/jpeg'
    elif file_ext == 'png':
        mime_type = 'image/png'
    elif file_ext == 'gif':
        mime_type = 'image/gif'
    else:
        mime_type = 'image/jpeg'  # Default fallback
    
    return image_bytes, mime_type


async def save_upload_backend(
    session_id: str,
    original_filename: str,
    file_bytes: bytes,
    mime_type: str
) -> str:
    artifact_id = f"{uuid.uuid4()}_{original_filename}"

    artifact_part = Part(
        inline_data=Blob(mime_type=mime_type, data=file_bytes),
        text=f"Original filename: {original_filename}"
    )

    # Save directly into artifact service under this session
    await artifact_service.save_artifact(
        session_id=session_id,
        filename=artifact_id,
        artifact=artifact_part,
        user_id=USER_ID,
        app_name=APP_NAME
    )

    return artifact_id

# await save_upload_as_artifact(context, "imgs.jpeg", bytesFile, mimeType)


async def before_model_include_artifact(callback_context: CallbackContext, llm_request: LlmRequest):
    artifact_list = await callback_context.list_artifacts()
    print(f"Listing Artifacts: {artifact_list}")

    if not artifact_list:
        return None

    # pick the artifact you want to send
    artifact_name = artifact_list[0]

    # load the saved artifact
    saved_part = await callback_context.load_artifact(filename=artifact_name)

    # extract the blob with image bytes from the Part
    if getattr(saved_part, "inline_data", None):
        blob = saved_part.inline_data
    else:
        print("Artifact has no inline_data; cannot send image")
        return None

    # inject actual image part into the LLM request
    for content in llm_request.contents:
        # 1. insert human-readable context
        content.parts.insert(0, Part(text=f"Here is the uploaded image: {artifact_name}"))

        # 2. insert the actual image bytes
        content.parts.insert(1, Part(inline_data=blob))
        
        print("content: '{content}'")

    print("Injected image binary into LLM request")
    return None

session_service = InMemorySessionService()


async def init_session(app_name:str,user_id:str,session_id:str) -> InMemorySessionService:
    global ARTIFACT_ID
    
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    [fileInBytes, fileMimeType] = load_image_as_binary()
    
    artifact_id = await save_upload_backend(
        session_id=SESSION_ID,
        original_filename="imgs.jpeg",
        file_bytes=fileInBytes,
        mime_type=fileMimeType
    )    
    
    ARTIFACT_ID = artifact_id 
    # print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")
    return session

session = asyncio.run(init_session(APP_NAME,USER_ID,SESSION_ID))

# Create agent after ARTIFACT_ID is set
photo_agent = Agent(
    name="photo_inspector_agent",
    model=MODEL_GEMINI_2_5_FLASH,
    instruction=(
         f"We have a user uploaded file referred to as {{artifact.{ARTIFACT_ID}}}. "
        "If you see a picture having 3 rows and 13 columns followed by some student details mentioned in the left most part, then its a picture of scorecard." 
        "Each column has rows like these as follows: "
        "116216(1) : This first row is having the paper code along with the total credits present in the bracket." 
        "25 | 50 : : This second row is a bifurcation of the internal and external marks." 
        "75 (A+) : This third row is the total marks and the grade that corresponds to it."
        "Now, based on these data, your ONLY task is to answer questions based on the image."
        "If image is not of scorecard format, then respectfully convey that to the user."
    ),
    before_model_callback=before_model_include_artifact
)
# print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
#   print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  print(f"<<< Agent Response: {final_response_text}")
  
  
# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.
runner = Runner(
    agent=photo_agent, # The agent we want to run
    app_name=APP_NAME,   # Associates runs with our app
    session_service=session_service, # Uses our session manager
    artifact_service=artifact_service
)
# print(f"Runner created for agent '{runner.agent.name}'.")

async def run_conversation():
    await call_agent_async("What is the SID and enrollment number present in this scorecard image and what is the total score of the subject with papercode '116210'?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESSION_ID)
    
    
if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")