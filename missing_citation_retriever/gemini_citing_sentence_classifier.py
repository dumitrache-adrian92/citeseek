import logging
from google import genai
from google.genai import types
import concurrent.futures
import time
import random


class GeminiCitingSentenceClassifier:
    """
    A class to classify sentences in paragraphs as check-worthy or non-check-worthy using the Gemini model.

    This class handles rate limits and retries using exponential backoff, and processes batches of paragraphs in
    parallel.

    Attributes:
        client (genai.Client): The GenAI client for interacting with the Gemini model.
        model (str): The model endpoint for the Gemini model.
        generate_content_config (types.GenerateContentConfig): Configuration for content generation, including safety
        settings.

    Methods:
        process_single_input(input_text: list[str], initial_delay: float = 1, max_retries: int = 5) -> list[bool] | str:
            Classify a single paragraph of text as check-worthy or non-check-worthy with exponential backoff.
        process_batch_parallel(batch: list[list[str]], initial_delay: float = 0.1, max_retries: int = 25, max_workers:
                int = 12) -> list[list[bool]]:
            Classify a batch of paragraphs in parallel, handling rate limits and retries.
    """
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project="438747908796",
            location="europe-west9",
        )

        self.model = "projects/438747908796/locations/europe-west9/endpoints/3205177550036795392"

        self.generate_content_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.1,
            max_output_tokens=8192,
            safety_settings=[types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )],
        )

    def process_single_input(self,
                             input_text: list[str],
                             initial_delay: float = 1,
                             max_retries: int = 5) -> list[bool] | str:
        """
        Classify a single paragraph of text as check-worthy or non-check-worthy with exponential backoff.
        Args:
            input_text (list[str]): A list of sentences that represent a paragraph.
            initial_delay (float): Initial delay before retrying in case of rate limit.
            max_retries (int): Maximum number of retries in case of rate limit or other errors.
        Returns:
            list[bool] | str: A list of booleans indicating whether each sentence is check-worthy or not,
                              or an error message if processing fails.
        """
        retry_count = 0
        delay = initial_delay

        while retry_count <= max_retries:
            try:
                result = self._classify(input_text)
                return result
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "API token" in str(e):
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = delay * (2 ** retry_count) + random.uniform(0, 1)
                        logging.warning(
                            f"Rate limit hit, backing off for {wait_time:.2f} seconds "
                            f"(retry {retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                        delay = wait_time
                    else:
                        logging.warning(f"Max retries exceeded for input, skipping")
                        return ""
                else:
                    logging.warning(f"Error processing input: {e}")
                    logging.warning(input_text)
                    retry_count += 1

        return "ERROR"

    def process_batch_parallel(self,
                               batch: list[list[str]],
                               initial_delay: float = 0.1,
                               max_retries: int = 25,
                               max_workers: int = 10) -> list[list[bool]]:
        """
        Classify a batch of paragraphs in parallel, handling rate limits and retries.

        Args:
            batch (list[list[str]]): A batch of paragraphs, each represented as a list of sentences.
            initial_delay (float): Initial delay before retrying in case of rate limit.
            max_retries (int): Maximum number of retries in case of rate limit or other errors.
            max_workers (int): Maximum number of parallel workers to use for processing the batch.

        Returns:
            list[list[bool]]: A list of results for each paragraph in the batch, where each result is either
        """
        def submit_with_delay(task_executor, fn, *args, **kwargs):
            time.sleep(random.uniform(0.1, 0.5))
            return task_executor.submit(fn, *args, **kwargs)

        logging.info(f"Classifying {len(batch)} paragraphs.")

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                submit_with_delay(executor, self.process_single_input, text, initial_delay, max_retries): i
                for i, text in enumerate(batch)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()

                    results.append((index, result))
                except Exception as e:
                    results.append((index, []))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _classify(self, paragraph: list[str]) -> list[bool]:
        """
        Handles classification of a paragraph of text as check-worthy or non-check-worthy using the Gemini model.

        Args:
            paragraph (list[str]): A list of sentences that represent a paragraph.
        Returns:
            list[bool]: A list of booleans indicating whether each sentence is check-worthy (True) or non-check-worthy
            (False).
        """
        def fix_pred(pred: list[str], length: int) -> list[str]:
            """
            Fix the prediction to ensure it has the correct length.
            If the prediction is shorter than the expected length, append '\nx: non-check-worthy'.
            If the prediction is longer, truncate it to the expected length.

            Args:
                pred (list[str]): The list of predictions from the model.
                length (int): The expected length of the prediction list.
            Returns:
                list[str]: The fixed prediction list with the correct length.
            """
            if len(pred) < length:
                return pred + ['x: non-check-worthy'] * (length - len(pred))
            if len(pred) > length:
                return pred[:length]  # Truncate if it's too long
            return pred

        input_text = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(paragraph)])

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=input_text
                    )
                ]
            )
        ]

        output = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.generate_content_config,
        )

        predictions = output.text.split("\n")

        if len(predictions) != len(paragraph):
            logging.warning(f"Warning: Expected {len(paragraph)} predictions, got {len(predictions)}. Adjusting "
                            f"predictions.")
            predictions = fix_pred(predictions, len(paragraph))

        predictions = [pred.split(":")[1].strip().lower() == "check-worthy" for pred in predictions]

        return predictions
