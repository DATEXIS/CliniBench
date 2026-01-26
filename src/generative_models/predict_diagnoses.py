import asyncio
import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from typing import List, Optional

import aiohttp
import httpx
import pandas as pd
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel, conlist, constr
from tqdm import tqdm
from tqdm.asyncio import tqdm, tqdm_asyncio

from src.generative_models.map_and_evaluate import short_code_from_code


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await tqdm.gather(*(sem_coro(c) for c in coros))


async def create_diagnoses_openai_client(prompts: List[str], system_prompt: str, labels, few_shots: Optional[List] = None):
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    # Create a custom httpx AsyncClient with higher limits
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=5000,  # Total connection pool size
            max_keepalive_connections=5000,  # Keepalive connections
        ),
        timeout=httpx.Timeout(6000.0),  # Optional: adjust timeout
    )
    client = AsyncOpenAI(
        base_url=args.vllm_ip + "/v1",
        api_key="-",
        http_client=http_client,
    )
    models = await client.models.list()
    model = models.data[0].id
    completions = []

    if os.getenv("DEBUG") == "1":
        prompts = prompts[:512]
        labels = labels[:512]

    # warmup request to not run the vllm instance OOM (helps to cache the compiled schema)
    warmups = []
    for i in range(3):
        warmups.append(client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"The admission note looks as follows: 1234",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnosis-list",
                    "schema": DiagnosisListRestrictedStr.model_json_schema(),
                },
            },
        ))
    await asyncio.gather(*warmups)

    for i, note in enumerate(prompts):
        completions.append(
            client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"The admission note looks as follows: {note}" if few_shots is None else f"The admission note looks as follows: {note} \n\n Similar patient notes look as follows: {few_shots[i].json()} \n\n Please specifiy the diagnoses for the given admission note following the style of diagnosis descriptions in the provided examples." ,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "diagnosis-list",
                        "schema": DiagnosisListRestrictedStr.model_json_schema(),
                    },
                }
            )
        )
    #resps = await asyncio.gather(*completions)
    resps = await gather_with_concurrency(5000, *completions)
    completions = [json.loads(x.choices[0].message.content)["diagnoses"] for x in resps]

    df = pd.DataFrame(
        {
            "results": completions,
            "prompts": prompts,
            "labels": labels,
            "icd_version": args.icd_version,
        },
        index=None,
    )
    df.to_parquet(args.results_file)




async def generate_diagnoses_apiserver(prompts):
    hostname = args.vllm_ip
    openai_api_key = "token-abc123"
    openai_api_base = f"{hostname}/v1"
    connected = False
    num_tries = 0
    while not connected:
        print("Num connection tries: ", num_tries)
        try:
            x = requests.get(f"{hostname}/health")
            if x.status_code == 200:
                connected = True
                time.sleep(5)
            else:
                time.sleep(5)
                num_tries += 1
        except:
            if num_tries > 500:
                raise RuntimeError("Could not connect to vllm server")
            time.sleep(5)
            num_tries += 1

    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=httpx.Timeout(None, read=None, write=None, connect=None),
    )

    models = await client.models.list()
    model = models.data[0].id
    completions = []
    timeout = aiohttp.ClientTimeout(
        total=1000000, connect=1000000, sock_connect=1000000, sock_read=1000000
    )

    MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2048"))
    BATCHSIZE = 256
    AIO_CONNECTION_LIMIT = int(os.getenv("AIO_CONNECTION_LIMIT", "2048"))

    logger.info("Start warmup request")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        resps = [
            session.post(
                url=f"{hostname}/v1/completions",
                json={
                    "prompt": "Test 1234456",
                    "model": model,
                    "echo": False,
                    "n": 1,
                    "temperature": 0,
                    "max_tokens": 32,
                    "stream": False,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "car-description",
                            "schema": DiagnosisListRestrictedStr.model_json_schema(),
                        },
                    },
                },
                headers={"Content-Type": "application/json"},
            )
            for _ in range(3)
        ]
        resps = await asyncio.gather(*resps)
        resps = [await x.json() for x in resps]

        logger.info("Warmup complete")

    if args.chain_of_thought:
        prompts = [
            x
            + "\n\n TASK: Solve this task step by step and give an explanation in maximum one or two sentences for each diagnosis decision."
            for x in prompts
        ]

        async with aiohttp.ClientSession(timeout=timeout) as session:
            cot_results = []
            for i in tqdm(
                range(0, len(prompts), BATCHSIZE),
                desc="Generate chain of thought answers",
            ):
                responses = []
                responses.append(
                    session.post(
                        url=f"{hostname}/v1/completions",
                        json={
                            "prompt": prompts[i : i + BATCHSIZE],
                            "model": model,
                            "echo": True,
                            "n": 1,
                            "temperature": 0,
                            "max_tokens": 256,
                            "stream": False,
                        },
                        headers={"Content-Type": "application/json"},
                    )
                )

                responses = await gather_with_concurrency(MAX_CONCURRENCY, *responses)
                cot_completions = [await x.json() for x in responses][0]["choices"]
                cot_results.extend([x["text"] for x in cot_completions])
            prompts = cot_results

            completions = []

            for i in tqdm(
                range(0, len(prompts), BATCHSIZE),
                desc="Generate diagnoses for chain-of-thought results",
            ):
                responses = []
                responses.append(
                    session.post(
                        url=f"{hostname}/v1/completions",
                        json={
                            "prompt": prompts[i : i + BATCHSIZE],
                            "model": model,
                            "echo": False,
                            "n": 1,
                            "temperature": 0,
                            "max_tokens": 1500,
                            "stream": False,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "car-description",
                                    "schema": DiagnosisListRestrictedStr.model_json_schema(),
                                },
                            },
                        },
                        headers={"Content-Type": "application/json"},
                    )
                )

                responses = await gather_with_concurrency(MAX_CONCURRENCY, *responses)
                t_completions = [await x.json() for x in responses][0]["choices"]
                completions.extend([x["text"] for x in t_completions])
    else:
        completions = []
        connector = aiohttp.TCPConnector(
            limit=AIO_CONNECTION_LIMIT, limit_per_host=AIO_CONNECTION_LIMIT
        )
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            responses = []

            for i in tqdm(range(0, len(prompts)), desc="Predict diagnoses"):
                resp = session.post(
                    url=f"{hostname}/v1/completions",
                    json={
                        "prompt": prompts[i],
                        "model": model,
                        "echo": False,
                        "n": 1,
                        "temperature": 0,
                        "max_tokens": 1500,
                        "stream": False,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "car-description",
                                "schema": DiagnosisListRestrictedStr.model_json_schema(),
                            },
                        },
                    },
                    headers={"Content-Type": "application/json"},
                )
                responses.append(resp)
            responses = await gather_with_concurrency(MAX_CONCURRENCY, *responses)
            print("Number of received responses: ", len(responses))
            t_completions = []
            for x in responses:
                try:
                    answer = await x.json()
                except Exception as e:
                    print("Error parsing answer from decoder api: ", e)
                    t_completions.append("NONE")
                try:
                    for choice in answer["choices"]:
                        try:
                            t_completions.append(choice["text"])
                        except Exception as e:
                            print("Exception: ", e)
                            result = "None"
                            t_completions.append(result)
                except Exception as error:
                    print("choices not found as key")
                    print("Exception: ", error)
                    print("Answer: ", answer)
                    t_completions.append("NONE")

            completions.extend(t_completions)
    print("Number of prompts: ", len(prompts))
    print("Number of completions: ", len(completions))
    results = []
    for c in completions:
        try:
            diag_list = DiagnosisListRestrictedStr.model_validate_json(c)
            results.append(diag_list)
            if len(diag_list.diagnoses) != args.num_predictions:
                print("Not enough predictions")
        except Exception as e:
            print("error parsing: ", c, " Exception: ", e)
            results.append(
                DiagnosisListRestrictedStr(
                    diagnoses=["NONE" for _ in range(args.num_predictions)]
                )
            )
    logger.info("Generation successful")

    return results, prompts


async def evaluate_cases(prompts, labels):
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    if os.getenv("DEBUG", "") == "1":
        prompts = prompts[:128]
        labels = labels[:128]

    results, prompts = await generate_diagnoses_apiserver(prompts)
    results = [x.diagnoses for x in results]

    df = pd.DataFrame(
        {
            "results": results,
            "prompts": prompts,
            "labels": labels,
            "icd_version": args.icd_version,
        },
        index=None,
    )
    df.to_parquet(args.results_file)
    logger.info("Saved dataframe to file")


def get_few_shot_candidates(
    few_shot_ids: pd.DataFrame,
    few_shot_notes: pd.DataFrame,
    num_candidates: int = 1,
):
    icd_9_descs = pd.read_csv(args.icd9_code_file)
    icd_9_descs = icd_9_descs.set_index("icd_code").to_dict()["long_title"]

    icd_10_descs = pd.read_csv(args.icd10_code_file)
    icd_10_descs = icd_10_descs.set_index("icd_code").to_dict()["long_title"]

    few_shot_candidates = {}
    for hadm_id, t_few_shots in tqdm(
        few_shot_ids.items(), desc="Map few shot labels to short descriptions"
    ):
        t_candidates = []
        for i in range(num_candidates):
            note = few_shot_notes.loc[t_few_shots[i]]
            if note["LONG_CODES"] is None:
                continue
            if args.icd_version == 9:
                note_labels = [
                    icd_9_descs[x] for x in note["LONG_CODES"] if x in icd_9_descs
                ]
            else:
                note_labels = [
                    icd_10_descs[x] for x in note["LONG_CODES"] if x in icd_10_descs
                ]
            cand = FewShotCandidate(diagnoses=note_labels, text=note["TEXT"])
            t_candidates.append(cand)
        few_shot_candidates[hadm_id] = FewShotCandidateList(candidates=t_candidates)

    return few_shot_candidates


def main():
    notes = pd.read_parquet(args.test_data)

    if args.few_shot:
        few_shot_ids = pd.read_parquet(args.few_shot_ids)
        few_shot_notes = pd.read_parquet(args.few_shot_note_data)

        ids = few_shot_ids[args.few_shot_column + "_few_shots"]
        hadm_ids = few_shot_ids[args.few_shot_column + "_hadm_id"]
        hadm_ids = hadm_ids[hadm_ids.notnull()].astype(int).astype(str).tolist()
        ids = ids[ids.notnull()].tolist()
        ids = [x.tolist() for x in ids]
        few_shot_data = {
            hadm_id: few_shots for hadm_id, few_shots in zip(hadm_ids, ids)
        }

        few_shot_candidates = get_few_shot_candidates(
            few_shot_data, few_shot_notes, num_candidates=args.num_few_shot_candidates
        )

    logger.info("Data loaded")
    cases = []
    labels = []
    few_shots = []
    notes = notes.dropna()
    for i, note in notes.iterrows():
        cases.append(note["TEXT"])
        if args.few_shot:
            few_shots.append(few_shot_candidates[str(note.name)])
        labels.append([short_code_from_code(x) for x in note["LONG_CODES"].tolist() if not pd.isna(note.TEXT)])
    prompts = []

    schema = str(DiagnosisListRestrictedStr.model_json_schema())
    SYSTEM_PROMPT = f"You are a medical professional. Given an admission note for a patient, present a list of possible diagnoses for the patient. Answer in the following schema: {schema}."

    if args.instruct:
        if args.few_shot:
            asyncio.run(create_diagnoses_openai_client(cases, SYSTEM_PROMPT, labels))
        else:
            asyncio.run(create_diagnoses_openai_client(cases, SYSTEM_PROMPT, labels, few_shots=few_shots))
    else:
        if args.few_shot:
            for i, note in enumerate(cases):
                task_prompt = f"You are a medical professional. Given an admission note for a patient, present a list of possible diagnoses for the patient. Similar patients look like this: {few_shots[i].json()}. The admission note is as follows: {note}. Give the diagnoses following the schema from the examples. Answer in the following schema: {schema} "
                prompts.append(task_prompt)
        else:
            for note in cases:
                task_prompt = f"You are a medical professional. Given an admission note for a patient, present a list of possible diagnoses for the patient. The admission note is as follows: {note}. Answer in the following schema: {schema} "
                prompts.append(task_prompt)
        logger.info("Prompts prepared")
        asyncio.run(evaluate_cases(prompts, labels))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--vllm_ip",
        default="http://localhost:8000",
        help="IP address and port of the vLLM inference server (e.g., 'http://localhost:8000')",
    )

    parser.add_argument(
        "--test_data",
        default="./data/preprocessed/icd-10/icu/test.parquet",
        help="Path to the CSV file containing the test dataset with admission notes to evaluate",
    )

    parser.add_argument(
        "--icd9_code_file",
        default="./data/icd9_codes.csv",
        help="Path to the CSV file containing all ICD-9 codes with their long descriptions",
    )

    parser.add_argument(
        "--icd10_code_file",
        default="./data/icd10_codes.csv",
        help="Path to the CSV file containing all ICD-10 codes with their long descriptions",
    )

    parser.add_argument(
        "--results_file",
        default="/tmp/results/results.json",
        help="Path where the evaluation results parquet file will be saved",
    )

    parser.add_argument(
        "--logfile",
        default="clinibench_logs.log",
        help="Path to the log file. Use different paths for parallel processes on shared filesystems to avoid conflicts.",
    )

    parser.add_argument(
        "--num_predictions",
        type=int,
        default=20,
        help="Maximum number of diagnosis predictions to generate per admission note",
    )

    parser.add_argument(
        "--min_diag_str_len",
        type=int,
        default=3,
        help="Minimum character length for each predicted diagnosis name (shorter predictions are filtered out)",
    )

    parser.add_argument(
        "--max_diag_str_len",
        type=int,
        default=70,
        help="Maximum character length for each predicted diagnosis name (longer predictions are truncated)",
    )

    parser.add_argument(
        "--icd_version",
        type=int,
        choices=[9, 10],
        default=10,
        help="ICD coding system version to use for evaluation (9 for ICD-9, 10 for ICD-10)",
    )

    parser.add_argument(
        "--chain_of_thought",
        default=False,
        action="store_true",
        help="Enable chain-of-thought prompting to have the model explain its reasoning before making predictions",
    )

    parser.add_argument(
        "--instruct",
        action="store_true",
        default=False,
        help="Indicate whether the model is an instruction tuned model (set the flag if the model was instruction tuned otherwise we assume the model is only a completion language model)",
    )

    parser.add_argument(
        "--few_shot",
        default=False,
        action="store_true",
        help="Enable few-shot learning by including example admission notes with diagnoses in the prompt",
    )

    parser.add_argument(
        "--num_few_shot_candidates",
        type=int,
        default=1,
        help="Number of example notes to include in the prompt (only used when --few_shot is enabled)",
    )

    parser.add_argument(
        "--few_shot_note_data",
        default="data/train_full.json",
        help="Path to the training dataset JSON file used for sampling few-shot example notes",
    )

    parser.add_argument(
        "--few_shot_ids",
        default="data/dev_few_shots.pq",
        help="Path to the Parquet file mapping each test admission note to its few-shot example candidates (by hadm_id)",
    )

    parser.add_argument(
        "--few_shot_column",
        choices=["icd9_hosp", "icd9_icu", "icd10_hosp", "icd10_icu"],
        default="icd9_hosp",
        help="Column name specifying which ICD codes to use from few-shot examples (should match the evaluation setting: hospital/ICU and ICD-9/10)",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    class DiagnosisListNormalStr(BaseModel):
        diagnoses: conlist(
            str, min_length=args.num_predictions, max_length=args.num_predictions
        )

    class DiagnosisListRestrictedStr(BaseModel):
        diagnoses: conlist(
            constr(
                min_length=args.min_diag_str_len if args.min_diag_str_len > 0 else None,
                max_length=args.max_diag_str_len,
            ),
            min_length=args.num_predictions,
            max_length=args.num_predictions,
        )

    class FewShotCandidate(BaseModel):
        text: str
        diagnoses: List[str]

    class FewShotCandidateList(BaseModel):
        candidates: List[FewShotCandidate]

    main()
