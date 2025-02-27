import pandas as pd
from q_loop import q_loop
from modified_tom_functions import getnotrawPrses


##it turns out that the real queue is faster than this loop, but for whatever reason it often doesn't work
def loop_generate_prs_matrix(test="m", duplicate_rows="mean", saveName=None, tailsTest="rightLeft",
                             random_shuffle_prsLoader=False, use_prsLoader=True, direction = False, onlyThesePrses = None):
    fundict = {}
    ###We also care about the column names
    if onlyThesePrses is None:
        if use_prsLoader:
            prses = getnotrawPrses()
        else:
            prses = pd.read_csv(
                "/net/mraid20/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv").set_index(
                "RegistrationCode").columns
    else:
        prses = onlyThesePrses
    ##each batch is one prs
    for prs_id in range(len(prses)):
        print("now starting prs number: ", prs_id, "/", str(len(prses)))
        # empty string matches positional argument for PRSpath, and the 0 fixes the prs id being an array
        fundict[prs_id] = q_loop(test, duplicate_rows, prses[prs_id], saveName, tailsTest, random_shuffle_prsLoader,
                                 use_prsLoader, prs_id, direction)  ##test can be "t" for t test or "r" for regression))
    for k, v in fundict.copy().items():  ##catch broken PRSes
        if v is None:
            del fundict[k]
    final_res = pd.concat(fundict.values(), axis=1)
    # final_res = final_res.loc[:,~final_res.columns.duplicated()] ##drop the duplicate indices we've accumulated atthis point
    print("Loader matrix finished!")
    return (final_res)