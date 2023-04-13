import numpy as np
import torch
import torch.nn as nn


def get_embedding(concept, clip_tokenizer, clip):
    """
    Compute the CLIP embedding of a given string.
    :param concept: the string to embed
    :param clip_tokenizer: the CLIP tokenizer
    :param clip: the CLIP model
    :return: the embedding vector as a numpy array
    """
    with torch.no_grad():
        inputs = clip_tokenizer(concept, return_tensors="pt").to(clip.device)
        embed = clip.get_text_features(**inputs)[0]

        # normalize the embedding to unit length
        embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
        return embed.cpu().numpy()


def cosine_similarity(image_embeds, text_embeds):
    """
    Compute the cosine similarity between image and text embeddings.
    :param image_embeds: the image embeddings
    :param text_embeds: the text embeddings
    :return: the cosine similarity
    """
    return image_embeds @ text_embeds.T


class MySafetyChecker(nn.Module):
    """
    A safety checker that uses CLIP to check if an image is safe, by comparing the image's embedding to a set of
    precomputed embeddings of bad concepts.
    """
    def __init__(self, clip, bad_embeddings):
        super().__init__()
        self.dtype = clip.dtype
        self.clip_model = clip
        self.concept_embeds = bad_embeddings

    @torch.no_grad()
    def forward(self, clip_input, images):
        image_embeds = self.clip_model.get_image_features(clip_input)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        image_embeds = image_embeds.cpu().float().numpy()
        sim = cosine_similarity(image_embeds, self.concept_embeds)
        print("similarity", sim)

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"concept_scores": {}, "bad_concepts": []}

            for concept_idx in range(len(sim[0])):
                concept_cos = sim[i][concept_idx]
                result_img["concept_scores"][concept_idx] = round(concept_cos, 3)

                # if the cosine similarity is above a threshold, the image is considered unsafe
                if result_img["concept_scores"][concept_idx] > 0.28:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            print(
                "Potential **BAD** content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts
