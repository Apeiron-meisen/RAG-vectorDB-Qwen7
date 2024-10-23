import json
from typing import Text, List, Optional, Dict


class JudgePrompter:
    def __init__(self, query: Text,
                 retrieved_documents: List,
                 vehicle_info: Text,
                 prompt_template):
        self.query = query
        self.vehicle_info = vehicle_info
        self.prompt_template = prompt_template
        self.prompt = None
        self.choice_mapping = {}
        self.retrieved_documents = retrieved_documents
        self.context_documents = []
        self.make_prompt()

    def make_prompt(self):
        self.context_documents = []
        chunks = ""
        choice_labels = [str(i + 1) for i in range(0, 20)]
        for i, retrieved_doc in enumerate(self.retrieved_documents):
            ans = f"{retrieved_doc['standard_question']}，{retrieved_doc['standard_answer']}"
            chunks += f"【{choice_labels[i]}】 {ans}\n\n"
            self.choice_mapping[choice_labels[i]] = retrieved_doc
            self.context_documents.append({"id": retrieved_doc["id"],
                                           "question": retrieved_doc['standard_question'],
                                           "answer": retrieved_doc["standard_answer"]})
            if len(chunks) > 4000:
                break
        self.choice_mapping["N"] = None
        content = {"query": self.query, "passage": chunks}

        self.prompt = self.prompt_template.format(content=json.dumps(content, ensure_ascii=False, indent=4))

    def get_document_by_choice(self, choice_label) -> Optional[Dict]:
        return self.choice_mapping.get(choice_label)
