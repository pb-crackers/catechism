import os
from dotenv import load_dotenv
import uuid
from supabase import create_client, Client

load_dotenv()
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(url, key)

"""
This file contains classes for conversation memory storage using Supabase.
"""

class ConversationMemory:
    """
    A class to handle storing conversation memory in Supabase.
    
    Each conversation record includes:
      - session_id: A unique identifier for the conversation (primary key).
      - summary: A summary of the conversation.
      - detailed_history: A complete record of the conversation.
    """

    def __init__(self, table_name: str = None):
        # Set the table name from argument or environment variable.
        self.table_name = table_name or os.environ.get("SUPABASE_TABLE_NAME", "Conversations")
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials (SUPABASE_URL and SERVICE_ROLE_KEY must be set).")
            
        # Initialize the Supabase client.
        self.supabase: Client = create_client(supabase_url, supabase_key)
        #print(f"Using Supabase table: {self.table_name}")

    def create_conversation(self, session_id: str, summary: str = "", last_user_input: str = "") -> bool:
        """
        Creates a new conversation record in Supabase.
        
        Args:
            session_id (str): Unique session identifier.
            summary (str): A summary of the conversation (default empty).
            last_user_input (str): Initial user input; used here as the initial detailed_history if provided.
        
        Returns:
            bool: True if creation was successful, False otherwise.
        """
        # Check if a conversation with the same session_id already exists.
        existing = self.get_conversation(session_id)
        if existing:
            #print(f"Conversation with session_id {session_id} already exists.")
            return False

        # Use last_user_input as the initial detailed_history if provided.
        detailed_history = last_user_input if last_user_input else ""

        data = {
            "id": session_id,
            "summary": summary,
            "detailed_history": detailed_history,
        }
        response = self.supabase.table(self.table_name).insert(data).execute()
        #print("create_conversation response:", response)
        return True

    def update_conversation(
        self,
        session_id: str,
        summary: str = None,
        last_user_input: str = None,
        detailed_history: str = None
    ):
        """
        Updates an existing conversation record in Supabase.

        Instead of overwriting the detailed_history, this method appends the new
        message to the existing detailed_history.

        Args:
            session_id (str): Unique session identifier.
            summary (str): New summary value (optional).
            last_user_input (str): New last user input value (optional, used if detailed_history is not provided).
            detailed_history (str): New detailed conversation message to append (optional).

        Returns:
            dict: The updated record (if successful) or None.
        """
        # First, retrieve the existing conversation to get its current detailed_history.
        current_record = self.get_conversation(session_id)
        if not current_record:
            #print(f"No conversation found for session_id: {session_id}")
            return None

        update_data = {}
        if summary is not None:
            update_data["summary"] = summary

        # Determine the new message to add.
        new_message = detailed_history if detailed_history is not None else last_user_input

        if new_message is not None:
            # Append the new message to the existing detailed_history.
            old_history = current_record.get("detailed_history", "")
            # If there's existing history, add a newline before the new message.
            if old_history:
                update_data["detailed_history"] = "\n" + new_message
            else:
                update_data["detailed_history"] = new_message

        if not update_data:
            #print("No update data provided.")
            return None

        response = self.supabase.table(self.table_name).update(update_data).eq("id", session_id).execute()
        #print("update_conversation response:", response)
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            return response.data[0]
        return None


    def get_conversation(self, session_id: str) -> dict:
        """
        Retrieves a conversation record by its session_id.
        
        Args:
            session_id (str): Unique session identifier.
        
        Returns:
            dict: The conversation record if found, or None.
        """
        response = self.supabase.table(self.table_name).select("*").eq("id", session_id).execute()
        #print("get_conversation response:", response)
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            return response.data[0]
        return None


class DevConversationMemory(ConversationMemory):
    """
    A class for development that uses a different table (default "Conversations_Dev")
    to store conversation memory in Supabase.
    """
    def __init__(self, table_name: str = None):
        table_name = table_name or os.environ.get("SUPABASE_DEV_TABLE_NAME", "Conversations_Dev")
        super().__init__(table_name)


# Example usage:
if __name__ == "__main__":
    db = ConversationMemory()
    test_session_id = str(uuid.uuid4())

    # Create a new conversation record.
    if db.create_conversation(test_session_id, summary="Initial summary", last_user_input="Hello, bot!"):
        print("Created conversation record.")

    # Retrieve the conversation record.
    conversation = db.get_conversation(test_session_id)
    print(conversation['detailed_history'])
    print("Retrieved conversation:", conversation)

    # Update the conversation record with new information.
    updated = db.update_conversation(test_session_id, summary="Updated summary", last_user_input="What can you do?")
    print("Updated conversation record:", updated)