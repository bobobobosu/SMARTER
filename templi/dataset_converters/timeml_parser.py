import os
import jsonlines
import json
from lxml import etree
from spacy.lang.en import English
from tqdm import tqdm


def parser(news):
    parsed = etree.XML(news)
    # print(etree.tostring(parsed, pretty_print=True))
    raw_text = str(parsed.xpath("string()"))

    # Extract sentences and its related annotations
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    sentences = [sent.string.strip() for sent in nlp(raw_text).sents]
    elements = filter(
        lambda x: x[0] is not None,
        sum([[(e.text, e), (e.tail, None)]
             for e in [parsed] + list(parsed)], []),
    )

    # create a dictionary {sentence:list_of_annotations}
    sentences_anno = {}
    sentences_idx = []
    for sentence in sentences:
        sentences_idx += [len(sentence.replace(" ", ""))]

    # populate the dictionary
    sentence = sentences.pop(0)
    cursor = 0
    for element in elements:
        annotation_len = len(element[0].replace(" ", ""))
        annotation_pos = (
            cursor,
            cursor + annotation_len,
        )  # position is index without counting spaces

        if not sentence in sentences_anno:
            sentences_anno[sentence] = {}
        if not element[1] is None:
            sentences_anno[sentence][annotation_pos] = element[1]

        cursor += annotation_len
        if cursor > sentences_idx[0]:
            cursor = cursor - sentences_idx.pop(0)
            sentence = sentences.pop(0)

    # extract all relations
    events_list = parsed.findall("EVENT")
    event_time = [event_handler(e, parsed) for e in events_list]

    relations_in_text = sum([i["temporal_relation"] for i in event_time], [])
    events_in_sentences = {}  # {sentence:{eid:pos}
    for k, v in sentences_anno.items():
        events_in_sentences[k] = {}
        for k1, v1 in v.items():
            if "eid" in v1.attrib:
                events_in_sentences[k][v1.attrib["eid"]] = k1
            if "tid" in v1.attrib:
                events_in_sentences[k][v1.attrib["tid"]] = k1

    relations_in_sentences = {}
    for k, v in events_in_sentences.items():
        related_relations = [
            i for i in relations_in_text if {i["lhs"], i["rhs"]}.issubset(v.keys())
        ]
        relations_in_sentences[k] = [
            {"lhs": v[i["lhs"]], "rhs": v[i["rhs"]], "rel": i["rel"]}
            for i in related_relations
        ]

    return relations_in_sentences


def event_handler(event, parsed):
    et_pair = {
        "event_name": event.text,
        "event_id": event.attrib["eid"],
        "temporal_relation": [],
        "event_relation": [],
    }

    ins_list = parsed.xpath(
        'MAKEINSTANCE[@eventID="{}"]'.format(et_pair["event_id"]))

    for ins in ins_list:
        eiid = ins.attrib["eiid"]
        et_pair["temporal_relation"] += tlink_handler(
            et_pair["event_id"], eiid, parsed)
        et_pair["event_relation"] += slink_handler(
            et_pair["event_id"], eiid, parsed)

    return et_pair


# tlink
def tlink_handler(eid, eiid, parsed):
    primary_text = parsed.xpath('EVENT[@eid="{}"]'.format(eid))[0].text
    tlinks = parsed.xpath('TLINK[@eventInstanceID="{}"]'.format(eiid))
    tlinks += parsed.xpath('TLINK[@relatedToEventInstance="{}"]'.format(eiid))
    temporal_relations = list()
    relations = []
    for tlink in tlinks:
        if "relatedToTime" in tlink.attrib or "timeID" in tlink.attrib:
            relate_tid = (
                tlink.attrib["relatedToTime"]
                if "relatedToTime" in tlink.attrib
                else tlink.attrib["timeID"]
            )
            relation_type = tlink.attrib["relType"]
            try:
                related_time_text = parsed.xpath(
                    'TIMEX3[@tid="{}"]'.format(relate_tid))[0].text
            except IndexError:
                continue
            relation_text = "{} {} {}".format(
                primary_text, relation_type, related_time_text)
            relations += [{"lhs": eid,
                           "rhs": relate_tid, "rel": relation_type}]

        elif "eventInstanceID" in tlink.attrib and tlink.attrib["eventInstanceID"] == eiid:
            relate_eiid = tlink.attrib["relatedToEventInstance"]
            relation_type = tlink.attrib["relType"]
            try:
                relate_eid = parsed.xpath('MAKEINSTANCE[@eiid="{}"]'.format(relate_eiid))[0].attrib[
                    "eventID"
                ]
            except IndexError:
                continue
            try:
                related_event_text = parsed.xpath(
                    'EVENT[@eid="{}"]'.format(relate_eid))[0].text
            except IndexError:
                continue
            relation_text = "{} {} {}".format(
                primary_text, relation_type, related_event_text)
            relations += [{"lhs": eid,
                           "rhs": relate_eid, "rel": relation_type}]

        elif (
            "relatedToEventInstance" in tlink.attrib
            and tlink.attrib["relatedToEventInstance"] == eiid
        ):
            relate_eiid = tlink.attrib["eventInstanceID"]
            relation_type = tlink.attrib["relType"]
            try:
                relate_eid = parsed.xpath('MAKEINSTANCE[@eiid="{}"]'.format(relate_eiid))[0].attrib[
                    "eventID"
                ]
            except IndexError:
                continue
            try:
                related_event_text = parsed.xpath(
                    'EVENT[@eid="{}"]'.format(relate_eid))[0].text
            except IndexError:
                continue
            relation_text = "{} {} {}".format(
                related_event_text, relation_type, primary_text)
            relations += [{"lhs": relate_eid,
                           "rhs": eid, "rel": relation_type}]
        temporal_relations.append(relation_text)
    return relations


# slink
def slink_handler(eid, eiid, parsed):
    primary_text = parsed.xpath('EVENT[@eid="{}"]'.format(eid))[0].text
    slinks = parsed.xpath('SLINK[@eventInstanceID="{}"]'.format(eiid))
    event_relations = list()
    for slink in slinks:
        sub_eid = parsed.xpath(
            'MAKEINSTANCE[@eiid="{}"]'.format(
                slink.attrib["subordinatedEventInstance"])
        )[0].attrib["eventID"]
        sub_text = parsed.xpath('EVENT[@eid="{}"]'.format(sub_eid))[0].text
        event_relations.append("{} to {}".format(primary_text, sub_text))
    return event_relations


if __name__ == "__main__":
    data_dir = "/mnt/AAI_Project/_Dataset/timebank_1_2/data"
    data_paths = ["timeml", "extra"]
    sentence_rels = {}
    for data_path in data_paths:
        file_dir = os.path.join(data_dir, data_path)
        file_list = os.listdir(file_dir)
        print("parse {}...".format(file_dir))
        for file in tqdm(file_list, desc='extracting sentence_rels'):
            with open(os.path.join(file_dir, file)) as f:
                news = f.read().replace("\n", "")
                sentence_rels = {**sentence_rels, **parser(news)}
    json.dump(sentence_rels, open('sentence_rels.json', "w"), indent=4)
    ddd = 9
