#!/usr/bin/python

# Implementation of Porter's algorithm
# (described at https://people.scs.carleton.ca/~armyunis/projects/KAPI/porter.pdf)

import re

word = ""

VOWEL = "(?:[aeiou]|(?:(?<=[aeiou])y))"
CONSONANT = "(?:[bcdfghjklmnpqrstvwxz]|(?:(?<![aeiou])y))"
CONSONANT_RULE1B_1 = "(?:[bcdfghjkmnpqrtvwx]|(?:(?<![aeiou])y))"
CONSONANT_RULE1B_2 = "(?:[bcdfghjklmnpqrstvz]|)"

def stem(word: str) -> str:
    word = word.lower()
    out = ""
    # Step 1a:
    #
    # SSES -> SS
    # IES -> I
    # SS -> SS
    # S -> (null)
    if re.match(r'(.+)sses$', word):
        out = re.sub(r'(.+)sses$', r'\1ss', word)
    elif re.match(r'(.+)ies$', word):
        out = re.sub(r'(.+)ies$', r'\1i', word)
    elif re.match(r'(.+)ss$', word):
        out = re.sub(r'(.+)ss$', r'\1ss', word)
    elif re.match(r'(.+)s$', word):
        out = re.sub(r'(.+)s$', r'\1', word)
    # Step 1b:
    # A sentence is structured as C*(VC){m}V*, where V represents one or more vowels and C represents one or more consonants
    # (m>0) EED -> EE
    # (*V*) ED -> (null)
    # (*V*) ING -> (null)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "eed$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "eed$", r'\1ee', word)
    elif re.match('(.*' + VOWEL + '.*)ed$', word) or re.match('(.*' + VOWEL + '.*)ing$', word):
        if re.match('(.*' + VOWEL + '.*)ed$', word):
            out = re.sub('(.*' + VOWEL + '.*)ed$', r'\1', word)
        if re.match('(.*' + VOWEL + '.*)ing$', word):
            out = re.sub('(.*' + VOWEL + '.*)ing$', r'\1', word)
        # If rule 2 or 3 of step 1b succeeds: 
        # AT -> ATE
        # BL -> BLE
        # IZ -> IZE
        # (*d and not (*L or *S or *Z)) -> single letter, where *d represents double consonants
        # (m=1 and *o) -> E, where *o signifies that the stem ends CVC, where the second C is not W, X or Y 
        out = re.sub(r'(.+)at$', r'\1ate', word)
        out = re.sub(r'(.+)bl$', r'\1ble', word)
        out = re.sub(r'(.+)iz$', r'\1ize', word)
        out = re.sub('(.+)' + '(' + CONSONANT_RULE1B_1 + ')' + '\2' + '$', r'\1\2', word)
        out = re.sub("(" + CONSONANT + "+" + "(?:" + VOWEL + "+" + CONSONANT_RULE1B_2 + "+" + ")", r'\1e', word)
    # Step 1c:
    # (*v*) Y -> I
    if re.match('(.*' + VOWEL + '.*)y$', word):
        out = re.sub('(.*' + VOWEL + '.*)y$', r'\1i', word)
    # Step 2:
    # (m>0) ATIONAL -> ATE
    # (m>0) TIONAL -> TION
    # (m>0) ENCI -> ENCE
    # (m>0) ANCI -> ANCE
    # (m>0) IZER -> IZE
    # (m>0) ABLI -> ABLE
    # (m>0) ALLI -> AL
    # (m>0) ENTLI -> ENT
    # (m>0) ELI -> E
    # (m>0) OUSLI -> OUS
    # (m>0) IZATION -> IZE
    # (m>0) ATION -> ATE
    # (m>0) ATOR -> ATE
    # (m>0) ALISM -> AL
    # (m>0) IVENESS -> IVE
    # (m>0) FULNESS -> FUL
    # (m>0) OUSNESS -> OUS
    # (m>0) ALITI -> AL
    # (m>0) IVITI -> IVE
    # (m>0) BILITI -> BLE
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ational$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ational$", r'\1ate', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "tional$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "tional$", r'\1tion', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "enci$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "enci$", r'\1ence', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "anci$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "anci$", r'\1ance', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "izer$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "izer$", r'\1ize', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "abli$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "abli$", r'\1able', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "alli$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "alli$", r'\1al', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "entli$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "entli$", r'\1ent', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "eli$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "eli$", r'\1e', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ousli$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ousli$", r'\ous', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ization$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ization$", r'\1ize', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ation$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ation$", r'\1ate', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ator$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ator$", r'\1ate', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "alism$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "alism$", r'\1al', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "iveness$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "iveness$", r'\1ive', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "fulness$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "fulness$", r'\1ful', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ousness$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ousness$", r'\1ous', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "aliti$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "aliti$", r'\1al', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "iviti$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "iviti$", r'\1ive', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ational$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "biliti$", r'\1ble', word)
    
    # Step 3:
    # (m>0) ICATE -> IC
    # (m>0) ATIVE -> (null)
    # (m>0) ALIZE -> AL
    # (m>0) ICITI -> IC
    # (m>0) ICAL -> IC
    # (m>0) FUL -> (null)
    # (m>0) NESS -> (null)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "icate$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "icate$", r'\1ic', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ative$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ative$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "alize$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "alize$", r'\1al', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "iciti$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "iciti$", r'\1ic', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ical$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ical$", r'\1ic', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ful$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ful$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ness$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "+" + VOWEL + "*" + ")" + "ness$", r'\1', word)

    # Step 4:
    # (m>1) AL -> (null)
    # (m>1) ANCE -> (null)
    # (m>1) ENCE -> (null)
    # (m>1) ER -> (null)
    # (m>1) IC -> (null)
    # (m>1) ABLE -> (null)
    # (m>1) IBLE -> (null)
    # (m>1) ANT -> (null)
    # (m>1) EMENT -> (null)
    # (m>1) MENT -> (null)
    # (m>1) ENT -> (null)
    # (m>1 and (*S or *T)) ION -> (null)
    # (m>1) OU -> (null)
    # (m>1) ISM -> (null)
    # (m>1) ATE -> (null)
    # (m>1) ITI -> (null)
    # (m>1) OUS -> (null)
    # (m>1) IVE -> (null)
    # (m>1) IZE -> (null)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "al$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "al$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ance$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ance$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ence$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ence$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "er$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "er$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ic$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ic$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "able$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "able$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ible$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ible$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ant$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ant$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ement$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ement$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ment$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ment$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ent$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ent$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + "s" + ")" + "ion$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + "s" + ")" + "ion$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + "t" + ")" + "ion$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + "t" + ")" + "ion$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ou$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ou$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ism$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ism$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ate$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ate$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "iti$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "iti$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ous$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ous$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ive$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ive$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ize$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "ize$", r'\1', word)
    
    # Step 5a:
    # (m>1) E -> (null)
    # (m=1 and not *o) E -> (null)
    
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "e$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + VOWEL + "*" + ")" + "e$", r'\1', word)
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT_RULE1B_2 + "+" + ")" + "{1}" + "*" + ")" + "e$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT_RULE1B_2 + "+" + ")" + "{1}" + "*" + ")" + "e$", r'\1', word)
    
    # Step 5b:
    # (m>1 and *d and *L) -> single letter
    if re.search("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + "ll" + ")" + "$", word):
        out = re.sub("(" + CONSONANT + "*" + "(?:" + VOWEL + "+" + CONSONANT + "+" + ")" + "{2,}" + "ll" + ")" + "$", r'\1l', word)
        
    return out