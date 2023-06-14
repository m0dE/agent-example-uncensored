import 'dotenv/config';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { LLamaEmbeddings } from "llama-node/dist/extensions/langchain.js";
import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationChain } from "langchain/chains";
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
  } from "langchain/prompts";
import prompt from "prompt";
import { BufferMemory } from "langchain/memory";

// const model = path.resolve(process.cwd(), "Wizard-Vicuna-13B-Uncensored.bin");
const model = new ChatOpenAI({ temperature: 0 });

const llama = new LLM(LLamaCpp);
const config = {
    path: model,
    enableLogging: true,
    nCtx: 1024,
    nParts: -1,
    seed: 0,
    f16Kv: false,
    logitsAll: false,
    vocabOnly: false,
    useMlock: false,
    embedding: true,
    useMmap: true,
    nGpuLayers: 0
};

const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
    ),
    new MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
]);


const chain = new ConversationChain({
    memory: new BufferMemory({ returnMessages: true, memoryKey: "history" }),
    prompt: chatPrompt,
    llm: model,
});

prompt.start();

function ask() {
    // Ask for name until user inputs "done"

    prompt.get(['input'], function(err, result) {
        getResponse(result.input)
    });
}

ask();

const getResponse = async (input) => {
    const response = await chain.call({
                        input: input
                        });
    console.log(response);
    ask();            
}




// const llama = new LLM(LLamaCpp);
// const config = {
//     path: model,
//     enableLogging: true,
//     nCtx: 1024,
//     nParts: -1,
//     seed: 0,
//     f16Kv: false,
//     logitsAll: false,
//     vocabOnly: false,
//     useMlock: false,
//     embedding: true,
//     useMmap: true,
//     nGpuLayers: 0
// };

// const run = async () => {
//     await llama.load(config);
//     // Load the docs into the vector store
//     const vectorStore = await MemoryVectorStore.fromTexts(["Hello world", "Bye bye", "hello nice world"], [{ id: 2 }, { id: 1 }, { id: 3 }], new LLamaEmbeddings({ maxConcurrency: 1 }, llama));
//     // Search for the most similar document
//     prompt.start();
    
//     function ask() {
//         // Ask for name until user inputs "done"

//         prompt.get(['input'], function(err, result) {
//             var resultOne = vectorStore.similaritySearch(result.input, 1)
//             console.log(resultOne);
//             ask();            
//         });
//     }
    
//     ask();    
// };

// run();