# =========================================================================
# Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os
import regex as re
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class LCSObject:
    """Class object to store a log group with the same template"""

    def __init__(self, logTemplate="", logIDL=[]):
        self.logTemplate = logTemplate
        self.logIDL = logIDL


class Node:
    """A node in prefix tree data structure"""

    def __init__(self, token="", templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()


class LogParser:
    """LogParser class

    Attributes
    ----------
        path : the path of the input file
        logName : the file name of the input file
        savePath : the path of the output file
        tau : how much percentage of tokens matched to merge a log message
    """

    def __init__(
        self,
        log_file,
        out_dir,
        log_format,
        tau=0.5,
        regex=[],
        keep_para=False,
        *args,
        **kwargs,
    ):
        self.log_file = log_file
        self.log_name = os.path.basename(log_file)
        self.out_dir = out_dir
        self.tau = tau
        self.logformat = log_format
        self.regex = regex
        self.keep_para = keep_para

    def LCS(self, seq1, seq2):
        lengths = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 - 1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2 - 1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1 - 1] == seq2[lenOfSeq2 - 1]
                result.insert(0, seq1[lenOfSeq1 - 1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def SimpleLoopMatch(self, logClustL, seq):
        for logClust in logClustL:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)
            if all(token in token_set or token == "<*>" for token in logClust.logTemplate):
                return logClust
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        retLogClust = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if childn.logClust is not None:
                    constLM = [w for w in childn.logClust.logTemplate if w != "<*>"]
                    if float(len(constLM)) >= self.tau * length:
                        return childn.logClust
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)

        return retLogClust

    def LCSMatch(self, logClustL, seq):
        retLogClust = None
        maxLen = -1
        maxClust = None
        set_seq = set(seq)
        size_seq = len(seq)
        for logClust in logClustL:
            set_template = set(logClust.logTemplate)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self.LCS(seq, logClust.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(logClust.logTemplate) < len(maxClust.logTemplate)):
                maxLen = len(lcs)
                maxClust = logClust

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append("<*>")
            if not lcs:
                break
        if i < len(seq):
            retVal.append("<*>")
        return retVal

    def addSeqToPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != "<*>"]

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
            # Do not Match
            else:
                parentn.childD[tokenInSeq] = Node(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.logClust is None:
            parentn.logClust = newCluster

    def removeSeqFromPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != "<*>"]

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def outputResult(self, logClustL):
        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []
        id_counter = 0

        for logclust in logClustL:
            template_str = " ".join(logclust.logTemplate)
            for logid in logclust.logIDL:
                templates[logid - 1] = template_str
                ids[logid - 1] = str(id_counter)
            df_event.append([str(id_counter), template_str, len(logclust.logIDL)])
            id_counter += 1

        df_event = pd.DataFrame(df_event, columns=["EventId", "EventTemplate", "Occurrences"])

        self.df_log["EventId"] = ids
        self.df_log["EventTemplate"] = templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.out_dir, self.log_name + "_structured.csv"), index=False)
        df_event.to_csv(os.path.join(self.out_dir, self.log_name + "_templates.csv"), index=False)

    def printTree(self, node, dep):
        pStr = ""
        for _ in range(dep):
            pStr += "\t"

        if node.token == "":
            pStr += "Root"
        else:
            pStr += node.token
            if node.logClust is not None:
                pStr += "-->" + " ".join(node.logClust.logTemplate)
        print(pStr + " (" + str(node.templateNo) + ")")

        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self):
        starttime = datetime.now()
        print('Parsing file:', self.log_file)
        self.load_data()
        rootNode = Node()
        logCluL = []

        # count = 0
        for line in self.df_log.itertuples():
            logID = line.LineId
            # logmessageL = list(filter(lambda x: x != "", re.split(r"[\s=:,]", self.preprocess(line.Content))))
            logmessageL = self.preprocess(line.Content).strip().split()
            constLogMessL = [w for w in logmessageL if w != "<*>"]

            # Find an existing matched log cluster
            matchCluster = self.PrefixTreeMatch(rootNode, constLogMessL, 0)

            if matchCluster is None:
                matchCluster = self.SimpleLoopMatch(logCluL, constLogMessL)

                if matchCluster is None:
                    matchCluster = self.LCSMatch(logCluL, logmessageL)

                    # Match no existing log cluster
                    if matchCluster is None:
                        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])
                        logCluL.append(newCluster)
                        self.addSeqToPrefixTree(rootNode, newCluster)
                    # Add the new log message to the existing cluster
                    else:
                        newTemplate = self.getTemplate(
                            self.LCS(logmessageL, matchCluster.logTemplate),
                            matchCluster.logTemplate,
                        )
                        if " ".join(newTemplate) != " ".join(matchCluster.logTemplate):
                            self.removeSeqFromPrefixTree(rootNode, matchCluster)
                            matchCluster.logTemplate = newTemplate
                            self.addSeqToPrefixTree(rootNode, matchCluster)
            if matchCluster:
                matchCluster.logIDL.append(logID)
            # count += 1
            # if count % 1000 == 0 or count == len(self.df_log):
            #     print("Processed {0:.1f}% of log lines.".format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.outputResult(logCluL)
        print("Parsing done. [Time taken: {!s}]".format(datetime.now() - starttime))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(self.log_file, regex, headers)

    def preprocess(self, line):
        for currentRex in self.regex:
            line = re.sub(currentRex, "<*>", line)
        return line

    def log_to_dataframe(self, log_file, regex, headers):
        """Function to transform log file to dataframe"""
        log_messages = []
        linecount = 0
        with open(log_file, 'r', errors='ignore') as fin:
            for line in tqdm(fin):
                line = re.sub(r"[^\x00-\x7F]+", "<NASCII>", line)
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                if linecount >= 10000000:
                    break

        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', [i for i in range(1, linecount + 1)])
        return logdf

    def generate_logformat_regex(self, logformat):
        """Function to generate regular expression to split log messages"""
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = (list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list])
        return parameter_list
