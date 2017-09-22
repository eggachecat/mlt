import pylab as plt


def draw_data(data, pause=False, class_index=2, style=None):
    """Function to draw Data
    Parameters
    ----------
    data : 3-d matrix-like object
       one column should be indicate the class of this row
    pause : 
    class_index : int
        the index of the class-column
    style : string-array object
        the styles for each class
    """
    index_set = [0, 1, 2]
    index_set.pop(class_index)

    if style is None:
        style = ["ro", "go", "bo", "r^", "g^", "b^", "bs", "rs", "gs"]

    category_style_dict = dict()
    style_ctr = 0

    for row in data:
        key = row[class_index]
        if not category_style_dict.get(key):
            try:
                category_style_dict[key] = style[style_ctr]
                style_ctr += 1
            except IndexError:
                print(
                    "The length of color-style-array smaller than the number of categories!! ")
                exit()

        cat_style = category_style_dict[key]
        plt.plot(row[index_set[0]], row[index_set[1]], cat_style)
    print("done")
    if pause:
        plt.pause(0)
