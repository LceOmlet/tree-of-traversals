import re

import requests
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from wikimapper import WikiMapper
import wptools
import wikipedia

from wikidata.label_desc_from_qids import query_label_and_descriptions
from wikidata.title_to_qid import titles_to_qids

import logging
logger = logging.getLogger(__name__)


mapper = WikiMapper("index_enwiki-latest.db")

extract_entities_prompt_template = """\
Query: {query}

Identify unique knowledge graph entities in the above query. Include only specific entities named in the query (people/places/organizations/events).
Add a 1-3 word disambiguation if necessary. The disambiguation should be clear from the query. Otherwise, leave disambiguation blank.


<< EXAMPLE >>
Query: What is the place of birth of the performer of song When I was Your Man?
Entities:
When I was Your Man - song

Query: What book series are Ron Weasley and Hermione Granger from?
Entities:
Ron Weasley - character of Harry Potter series
Hermione Granger - character from the Harry Potter stories

Query: When did Jean Martin's husband die?
Jean Martin - 

Query: "How many languages are widely used in India?"
Entities:
India - country

Query: "Who is older Leonardo DiCaprio or George Clooney?"
Entities:
Leonardo DiCaprio - actor
George Clooney - actor

Query: "Where was the director of film Batman (1989) born?"
Entities:
Batman - a 1989 film

<< END EXAMPLE >>


Fomatting rule: ```Entities:\n Separate the entity label and description with a ' - ', the short explaination on the right of ' - ' can be ormited if doesn't exist```

Please strictly and directly prompt out the results formated as the **formmating rule** describes after thinking like the above listed results.

We are not allowed to answer the question directly, instead you should extract entities in the problem for further retrieval. 

We should prompt simple and concise entities that are strictly the conditions in the query rather than any maked up entities. If you are not certain/sure, when you use the word 'maybe' or 'perhaps', for something or find your self repeat too much, please simply prompt one or few of the the most simple or general relations as posible.

Query: "{query}"

"""

def parse_elements(text):
    elements = text.split('\n')
    elements = list(filter(lambda x: x.strip() != '', elements))
    return elements


def name_to_id(name):
    # page = wikipedia.page(name)
    # url = page.url
    wikidata_id = mapper.title_to_id(name.replace(' ', '_'))
    return wikidata_id


def get_candidate_wikidata_entities(term):
    # search wikipedia and link
    if len(term) >= 300:
        logger.info(f"Search term ({term}) is too long")
        return [], []
    results = wikipedia.search(term, results=20)
    wikidata_ids = []
    final_results = []

    titles_to_qid = titles_to_qids(results)
    for r in results:
        if r not in titles_to_qid:
            logger.info(f"Could not match title {r} to QID")
            continue
        wikidata_ids.append(titles_to_qid[r])
        final_results.append(r)

    # search wikidata directly
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': term,
        'language': 'en'
    }
    data = requests.get(url, params=params).json()
    data = [(d['label'], d['id']) for d in data['search']]
    for label, qid in data:
        if qid not in wikidata_ids:
            wikidata_ids.append(qid)
            final_results.append(label)

    return final_results, wikidata_ids


def get_page(qid):
    page = wptools.page(wikibase=qid, silent=True)  # can search different wikimedia databases
    page = page.get_wikidata(show=False)  # can set up local version to make it faster
    return page


def get_all_candidate_qids(entities):
    ent_qids = []
    for ent in entities:
        try:
            label, description = ent.split(' - ')
            if not label.strip() and description.strip():
                label = description
                description = ''
        except ValueError:
            label = ent
            description = ""
        results, wikidata_ids = get_candidate_wikidata_entities(label)  # TODO: Only searches entities that have wikipedia pages
        ent_qids.append(wikidata_ids)
    all_qids = set(id for list_of_ids in ent_qids for id in list_of_ids)
    return ent_qids, all_qids


match_prompt_template = "Knowledge graph entities:\n{options}\n\n Choose which of the above entities best matches the following. Do not select list pages or disambiguation pages. Select an entity by index number based on matching entity label. The disambiguations may not be correct, in which case select the closest entity to the label. If there is no entity that could be a match say 'None'. \n\nEntity to match: {text}\nBased on the given entity label, the entity index that best matches is:"


def match_entity(context, ent, candidates, llm):
    options = [f"{i + 1}: {lab_desc}" for i, lab_desc in enumerate(candidates)]
    options = '\n'.join(options)
    match_prompt = match_prompt_template.format(**{'context': context, 'options': options, 'text': ent})
    logger.info(match_prompt)
    result = llm(match_prompt, n=1, stop="\n\n", max_token= 4096)
    try:
        best_match_idx = int(re.search("\d+", result[0])[0]) - 1
        logger.info(f"Matched:\n\t{ent}\n\t{candidates[best_match_idx]}")
    except ValueError:
        logger.info(f"No match found: {result[0]}")
        best_match_idx = None
    except IndexError:
        logger.info(f"Invalid option selected: {result[0]}")
        best_match_idx = None
    except TypeError:
        logger.info(f"Invalid option selected: {result[0]}")
        best_match_idx = None

    return best_match_idx


def match_entities_wikidata(query, entities, model):
    ent_qids, all_qids = get_all_candidate_qids(entities)
    qid_results = query_label_and_descriptions(all_qids)

    matched_wd_entities = {}
    wd2ent = {}
    unmatched_entities = []
    for i, e in enumerate(entities):
        options = []
        options_qid = []
        for qid in ent_qids[i]:
            try:
                options.append(f"{qid_results[qid]['label']} ({qid_results[qid]['description']})")
                options_qid.append(qid)
            except KeyError:
                continue
                # logger.info(f"could not append results for {qid}")
        idx = match_entity(query, e, options, model)
        if idx is None:
            unmatched_entities.append(e)
            continue
        matched_qid = options_qid[idx]
        # page = get_page(matched_qid)
        wd2ent[matched_qid] = e
        matched_wd_entities[matched_qid] = {
            'label': qid_results[matched_qid]['label'],
            'description': qid_results[matched_qid]['description'],
            'wikidata': matched_qid
        }
    return matched_wd_entities, wd2ent, unmatched_entities

# matched_entities, matched_idx = match_entities(data['entities'])
#
# query_label_and_descriptions
