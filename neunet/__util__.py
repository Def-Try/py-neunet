import time

"""
progress bar generator.

arg - things_done - number of things done (int)
arg - things_total - number of things total (int)
arg - width - width of progress bar - 40 (int)
arg - eta - generate eta time? - False (boolean)
arg - time_start - task start time, used for eta - None (int)
return - Progress bar (string)
"""
def _progress_bar(things_done, things_total, width=40, eta=False, time_start=None):
    if things_done == 0: things_done=1
    progress = things_done / things_total
    progress_width = int(width * progress)
    bar = str(things_done) + "/" + str(things_total) + " [" + "=" * progress_width + ">" + "-" * (width - progress_width) + "]"
    if eta:
        time_elapsed = time.time() - time_start
        time_remaining = round(time_elapsed * (things_total - things_done) / things_done)

        time_string = ""
        if time_remaining > 60:
            time_string = str(time_remaining // 60) + "m " + str(time_remaining % 60) + "s"
        else:
            time_string = str(time_remaining) + "s"
        eta_str = " ETA: {}".format(time_string)
    else:
        eta_str = ""
    return bar + eta_str
