import openai
import json

# Replace YOUR_API_KEY with your actual OpenAI API key
openai.api_key = 'YOUR_API_KEY'
openai.

def receive_event(event):
    # Process each event received from the stream
    if 'data' in event:
        data = json.loads(event['data'])
        if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            if 'message' in choice:
                print(f"Received: {choice['message']['content']}")

def open_stream():
    # Open a streaming session and send an initial message
    stream = openai.ChatCompletion.create_stream(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."}
        ],
        iter_callback=receive_event
    )
    
    try:
        for event in stream.events():
            receive_event(event)
    except KeyboardInterrupt:
        # Gracefully close the stream if the script is stopped
        print("Closing stream.")
        stream.close()

if __name__ == "__main__":
    open_stream()