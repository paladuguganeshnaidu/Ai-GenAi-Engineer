import OpenAI from "openai";
import readline from "readline";

const API_KEY = process.env.OPENROUTER_API_KEY;
const MODEL_NAME = "anthropic/claude-sonnet-5";

if (!API_KEY) {
  throw new Error("OPENROUTER_API_KEY is not set");
}

const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: API_KEY,
  defaultHeaders: {
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "Node Test Script",
  },
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log(`--- Interactive Terminal Chat (${MODEL_NAME}) ---`);
console.log("Type 'exit' or 'quit' to stop.\n");

function askPrompt() {
  rl.question("Prompt: ", async (userInput) => {
    const trimmedInput = userInput.trim();

    if (trimmedInput.toLowerCase() === "exit" || trimmedInput.toLowerCase() === "quit") {
      console.log("Exiting chat. Goodbye!");
      rl.close();
      return;
    }

    if (!trimmedInput) {
      askPrompt();
      return;
    }

    try {
      const completion = await openai.chat.completions.create({
        model: MODEL_NAME,
        max_tokens: 100, // Keeps token usage low to fit your account credit limit
        messages: [
          {
            role: "user",
            content: trimmedInput,
          },
        ],
      });

      const reply = completion.choices[0].message.content;

      console.log(`Model: ${MODEL_NAME}`);
      console.log(`Prompt: ${trimmedInput}`);
      console.log(`Response: ${reply}\n`);
    } catch (error) {
      console.error("Error communicating with OpenRouter:", error.message || error);
    }

    askPrompt();
  });
}

askPrompt();