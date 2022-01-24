#!/usr/bin/python3.7

import numpy as np
from .grammar import PathGrammar
from .length_model import PoissonModel
import glob
import re

# Viterbi decoding
class Frame_Based_Set_Viterbi(object):

    ### helper structure ###
    class TracebackNode(object):
        def __init__(self, start_frame, segment_length, label, predecessor):
            self.start_frame = start_frame
            self.segment_length = segment_length
            self.label = label
            self.predecessor = predecessor

    ### helper structure ###
    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, traceback):
                self.score = score
                self.traceback = traceback
        def update(self, key, score, traceback):
            if (not key in self) or (self[key].score <= score):
                self[key] = self.Hypothesis(score, traceback)

    # @grammar: the grammar to use, must inherit from class Grammar
    # @length_model: the length model to use, must inherit from class LengthModel
    # @frame_sampling: generate hypotheses every frame_sampling frames
    # @max_hypotheses: maximal number of hypotheses. Smaller values result in stronger pruning
    def __init__(self, grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf):
        self.grammar = grammar
        self.length_model = length_model
        self.frame_sampling = frame_sampling
        self.max_hypotheses = max_hypotheses

    # Viterbi decoding of a sequence
    # @log_frame_probs: logarithmized frame probabilities
    #                   (usually log(network_output) - log(prior) - max_val, where max_val ensures negativity of all log scores)
    # @return: the score of the best sequence,
    #          the corresponding framewise labels (len(labels) = len(sequence))
    #          and the inferred segments in the form (label, length)
    def decode(self, log_frame_probs, frame_list, label_list, scale):        
        self.n_anchors = len(label_list)
        assert self.n_anchors == len(frame_list)
        
        self.frame_list = frame_list
        self.label_list = label_list
        self.scale = scale

        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores allow for quick lookup if frame_sampling > 1
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        # decode each following time step
        current_id = 1
        while current_id < self.n_anchors - 1:
            for t in range(self.frame_sampling - 1, frame_scores.shape[0] - self.frame_sampling, self.frame_sampling):

                if t < self.frame_list[current_id] or t >= self.frame_list[current_id+1]:
                    continue
                hyps = self.decode_frame(t, hyps, frame_scores, current_id)
                
            current_id += 1
        
        # transition to end symbol
        final_hyp = self.finalize_decoding(frame_scores.shape[0] - 1, hyps, frame_scores, current_id)
        final_score, labels, segments = self.traceback(final_hyp, frame_scores.shape[0], frame_scores)
        return final_score, labels, segments


    ### helper functions ###
    def prune(self, hyps):
        if len(hyps) > self.max_hypotheses:
            tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )
            del_keys = [ x[1] for x in tmp[0 : -self.max_hypotheses] ]
            for key in del_keys:
                del hyps[key]

    def init_decoding(self, frame_scores):
        hyps = self.HypDict()        
        current_id = 0
        label = self.label_list[current_id]     
        for t in range(self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            if t < self.frame_list[current_id] or t >= self.frame_list[current_id+1]:
                continue
            key = (t, current_id)
            score = frame_scores[t, label] + self.length_model.score(t, label)
            hyps.update(key, score, estimated_total_length, self.TracebackNode(0, t + 1, label, None))
        return hyps

    def decode_frame(self, t, old_hyps, frame_scores, current_id):
        new_hyps = self.HypDict()  
        for key, hyp in list(old_hyps.items()):
            new_hyps[key] = hyp
            total_length, idx = key[0], key[1]
            length = t - total_length

            if idx + 1 != current_id or length <= 0:
                continue

            new_label = self.label_list[current_id]
            new_key = (t, current_id)
            segment_score = frame_scores[t, new_label] - frame_scores[total_length, new_label]
            score = hyp.score + segment_score + self.length_model.score(length, new_label)
            new_hyps.update(new_key, score, hyp.estimated_total_length + self.length_model.mean_lengths[new_label], self.TracebackNode(total_length+1, length, new_label, hyp.traceback))

        return new_hyps

    def finalize_decoding(self, t, old_hyp, frame_scores, current_id):
        final_hyp = self.HypDict.Hypothesis(-np.inf, t, None)
        new_label = self.label_list[current_id]
        for key, hyp in list(old_hyp.items()):
            total_length, idx = key[0], key[1]
            length = t - total_length

            if length <=0 or self.frame_list[-1] > t or idx + 1 != current_id:
                continue
            segment_score = frame_scores[t, new_label] - frame_scores[total_length, new_label]
            score = hyp.score + segment_score + self.length_model.score(length, new_label)
            if score > final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, self.TracebackNode(total_length+1, length, new_label, hyp.traceback)
        # return final hypothesis
        return final_hyp

    def traceback(self, hyp, n_frames, frame_scores):
        class Segment(object):
            def __init__(self, start_frame, label, length=0):
                self.start_frame, self.label, self.length = start_frame, label, length
                
        final_score = hyp.score
                
        action_set = self.grammar.action_set
        n_classes = self.grammar.n_classes()
        traceback = hyp.traceback
#         print('n_classes', n_classes)
        transcript = np.zeros(n_classes)
        segment_dict = dict()
        labels = []
        segments = []
                
        while not traceback == None:
            segment = Segment(traceback.start_frame, traceback.label)
            segments.append(segment)
            segments[-1].length = traceback.segment_length
            labels += [traceback.label] * segments[-1].length            
            
            if transcript[traceback.label] == 0:            
                segment_dict[traceback.label] = []
            segment_dict[traceback.label].append(segment)                
            transcript[traceback.label] += 1
            traceback = traceback.predecessor
            
        labels, segments = list(reversed(labels)), list(reversed(segments))
                            
        return final_score, labels, segments
