// index.mjs
import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";
console.log("imported everything")
const model = path.resolve(process.cwd(), "Wizard-Vicuna-13B-Uncensored.bin");
console.log("loaded model")

const llama = new LLM(LLamaCpp);
console.log("initiated LLM")

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
    embedding: false,
    useMmap: true,
};

const template = `How are you?`;
const prompt = `A chat between a user and an assistant.
USER: ${template}
ASSISTANT:`;

const run = async () => {
    await llama.load(config);
    console.log("loaded LLM config")

    await llama.createCompletion({
        nThreads: 4,
        nTokPredict: 2048,
        topK: 40,
        topP: 0.1,
        temp: 0.2,
        repeatPenalty: 1,
        prompt,
    }, (response) => {
        process.stdout.write(response.token);
    });
}

run();