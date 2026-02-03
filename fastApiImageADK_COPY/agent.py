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
from google.genai.types import Part, Blob
import uuid
from dotenv import load_dotenv
load_dotenv()

# Export these items for use in other modules
__all__ = ['call_agent_async']

APP_NAME = "scorecard_reader_app"
USER_ID = "user_1"
SESSION_ID = "session_001" 

MODEL_GEMINI_2_5_FLASH = "gemini-2.5-flash"

session_service = InMemorySessionService()
_session = None
_runner = None

async def _ensure_initialized():
    """Ensure session and runner are initialized (lazy initialization)."""
    global _session, _runner
    
    if _session is None:
        _session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
    
    if _runner is None:
        photo_agent = Agent(
            name="photo_inspector_agent",
            model=MODEL_GEMINI_2_5_FLASH,
            instruction=(
                f"We have a user uploaded file whose blob is shared in user's prompt which starts just after 'imageBlobStart imageBlob=(actual blob sent by user) imageBlobEnd'. "
                "If you see a picture having 3 rows and 13 columns followed by some student details mentioned in the left most part, then its a picture of scorecard." 
                "Each column has rows like these as follows: "
                "116216(1) : This first row is having the paper code along with the total credits present in the bracket." 
                "25 | 50 : : This second row is a bifurcation of the internal and external marks." 
                "75 (A+) : This third row is the total marks and the grade that corresponds to it."
                "Now, based on these data, your ONLY task is to answer questions based on the image."
                "If image is not of scorecard format, then respectfully convey that to the user."
            ),
        )
        
        _runner = Runner(
            agent=photo_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
    
    return _runner

async def call_agent_async(query: str, base64_blob: str):
  """Sends a query to the agent and prints the final response."""
  
  # Ensure runner is initialized
  runner = await _ensure_initialized()
  
  # Prepare the user's message with image blob
  content = types.Content(
      role='user', 
      parts=[
          types.Part(text=f"\n\n{query}"),
          types.Part(text="imageBlobStart imageBlob="),
          types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64_blob)),
          types.Part(text=" imageBlobEnd")
      ]
  )

  final_response_text = "Agent did not produce a final response." # Default

  async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          break # Stop processing events once the final response is found

  print(f"<<< Agent Response: {final_response_text}")
  return final_response_text