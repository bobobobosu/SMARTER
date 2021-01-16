import os
import jsonlines
import json
from lxml import etree
from spacy.lang.en import English
from tqdm import tqdm
from functools import reduce


def parser(news, sentences_in_one_data = 1):
    parsed = etree.XML(news)
    # hardcore for DCT
    # print(etree.tostring(parsed, pretty_print=True))
    raw_text = str(parsed.xpath("string()"))
    # print(raw_text)
    # Extract sentences and its related annotations
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    sentences = [sent.string for sent in nlp(raw_text).sents]
    # concat sentences for one data
    sentences = [''.join(sentences[i:i+sentences_in_one_data]) for i in range(len(sentences))]

    # sentencizer shouldn't drop any characters
    # assert raw_text == "".join([i.string for i in nlp(raw_text).sents])

    def recursive_tag_extractor(doc):
        return (
            [(doc.text, doc)]
            + sum([recursive_tag_extractor(i) for i in doc], [])
            + [(doc.tail, None)]
        )

    # extract (text, tag) pairs
    all_tags_with_text = list(filter(lambda x: x[0], recursive_tag_extractor(parsed)))

    # check extraction integrity
    # assert raw_text == "".join([i[0] for i in all_tags_with_text])

    # absolute_idx_of_tags
    absolute_idx_of_tags = reduce(
        lambda x, y: (x[0] + [(y, x[1])], x[1] + len(y[0])), all_tags_with_text, ([], 0)
    )[0]
    absolute_idx_of_tags = list(
        filter(lambda x: x[0][1] is not None, absolute_idx_of_tags)
    )


    # absolute_idx_of_sentences
    absolute_idx_of_sentences = reduce(
        lambda x, y: (x[0] + [(y, x[1])], x[1] + len(y)), sentences, ([], 0)
    )[0]

    # generate sentences_anno
    sentences_anno = {}
    for absolute_idx_of_sentence in absolute_idx_of_sentences:
        sentence = absolute_idx_of_sentence[0]
        abs_startidx = absolute_idx_of_sentence[1]
        abs_endidx = absolute_idx_of_sentence[1] + len(sentence)
        def spaceless_rng(idx_of_tag):
            startidx = len(sentence[:idx_of_tag[1]-abs_startidx+1].replace(" ",""))
            endidx = len(sentence[:idx_of_tag[1]+len(idx_of_tag[0][0])-abs_startidx+1].replace(" ",""))
            return f"{str(startidx)}_{str(endidx)}"
        sentences_anno[sentence] = {spaceless_rng(i):i[0][1] for i in absolute_idx_of_tags if abs_startidx <= i[1] <= abs_endidx}

    # extract all relations
    events_list = parsed.findall("EVENT")
    event_time = [event_handler(e, parsed) for e in events_list]

    relations_in_text = sum([i["temporal_relation"] for i in event_time], [])
    events_in_sentences = {}  # {sentence:{eid:pos}
    id_text = {}
    for k, v in sentences_anno.items():
        events_in_sentences[k] = {}
        for k1, v1 in v.items():
            if "eid" in v1.attrib:
                events_in_sentences[k][v1.attrib["eid"]] = k1
                id_text[v1.attrib["eid"]] = v1.text
            if "tid" in v1.attrib:
                events_in_sentences[k][v1.attrib["tid"]] = k1
                id_text[v1.attrib["tid"]] = v1.text

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

    ins_list = parsed.xpath('MAKEINSTANCE[@eventID="{}"]'.format(et_pair["event_id"]))

    for ins in ins_list:
        eiid = ins.attrib["eiid"]
        et_pair["temporal_relation"] += tlink_handler(et_pair["event_id"], eiid, parsed)
        et_pair["event_relation"] += slink_handler(et_pair["event_id"], eiid, parsed)

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
                    'TIMEX3[@tid="{}"]'.format(relate_tid)
                )[0].text
            except IndexError:
                continue
            relation_text = "{} {} {}".format(
                primary_text, relation_type, related_time_text
            )
            relations += [{"lhs": eid, "rhs": relate_tid, "rel": relation_type}]

        elif (
            "eventInstanceID" in tlink.attrib
            and tlink.attrib["eventInstanceID"] == eiid
        ):
            relate_eiid = tlink.attrib["relatedToEventInstance"]
            relation_type = tlink.attrib["relType"]
            try:
                relate_eid = parsed.xpath(
                    'MAKEINSTANCE[@eiid="{}"]'.format(relate_eiid)
                )[0].attrib["eventID"]
            except IndexError:
                continue
            try:
                related_event_text = parsed.xpath(
                    'EVENT[@eid="{}"]'.format(relate_eid)
                )[0].text
            except IndexError:
                continue
            relation_text = "{} {} {}".format(
                primary_text, relation_type, related_event_text
            )
            relations += [{"lhs": eid, "rhs": relate_eid, "rel": relation_type}]

        elif (
            "relatedToEventInstance" in tlink.attrib
            and tlink.attrib["relatedToEventInstance"] == eiid
        ):
            relate_eiid = tlink.attrib["eventInstanceID"]
            relation_type = tlink.attrib["relType"]
            try:
                relate_eid = parsed.xpath(
                    'MAKEINSTANCE[@eiid="{}"]'.format(relate_eiid)
                )[0].attrib["eventID"]
            except IndexError:
                continue
            try:
                related_event_text = parsed.xpath(
                    'EVENT[@eid="{}"]'.format(relate_eid)
                )[0].text
            except IndexError:
                continue
            relation_text = "{} {} {}".format(
                related_event_text, relation_type, primary_text
            )
            relations += [{"lhs": relate_eid, "rhs": eid, "rel": relation_type}]
        temporal_relations.append(relation_text)
    return relations


# slink
def slink_handler(eid, eiid, parsed):
    primary_text = parsed.xpath('EVENT[@eid="{}"]'.format(eid))[0].text
    slinks = parsed.xpath('SLINK[@eventInstanceID="{}"]'.format(eiid))
    event_relations = list()
    for slink in slinks:
        sub_eid = parsed.xpath(
            'MAKEINSTANCE[@eiid="{}"]'.format(slink.attrib["subordinatedEventInstance"])
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
        for file in tqdm(file_list, desc="extracting sentence_rels"):
            with open(os.path.join(file_dir, file)) as f:
                news = f.read().replace("\n", "")
                sentence_rels = {**sentence_rels, **parser(news)}
    json.dump(sentence_rels, open("sentence_rels.json", "w"), indent=4)
    ddd = 9
