# -- coding: utf-8 --
import json

Note=open('tags.txt',mode='w',encoding='utf-8')

with open('msra_train.txt', encoding='utf-8') as file:
    content = file.readlines()
    for line in content:
        line = json.loads(line.rstrip())
        # print(line['text'])
        # print(line['entity_list'])
        tags = ""
        if line['entity_list'] == []:
            tags = "O " * len(line['text'])
        else:
            line_len = len(line['text'])
            tags_len = 0
            for index, entity in enumerate(line['entity_list']):
                if index == 0:
                    tags += "O " * int(entity['entity_index']['begin'])
                    tags_len += int(entity['entity_index']['begin'])
                else:
                    pre_entity = line['entity_list'][index - 1]
                    tags += "O " * (int(entity['entity_index']['begin']) - int(pre_entity['entity_index']['end']))
                    tags_len += int(entity['entity_index']['begin']) - int(pre_entity['entity_index']['end'])
                tags += (entity['entity_type'] + " ") * (
                        int(entity['entity_index']['end']) - int(entity['entity_index']['begin']))
                tags_len += int(entity['entity_index']['end']) - int(entity['entity_index']['begin'])
            tags += "O " * (line_len - tags_len)
        # print(tags)
        data2 = json.dumps({'text':line['text'], 'tags':tags},ensure_ascii=False)
        print(data2)
        Note.write(data2+'\n')

Note.close()

