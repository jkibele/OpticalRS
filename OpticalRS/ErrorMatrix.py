import numpy as np
import csv, itertools

# This was copied from my acc-assess plugin (https://github.com/jkibele/acc-assess)
# ...but it's had a lot added to it since then.

#### TEST DATA ####------------------------------------------------------------------------
def congalton_table1():
    """
    Error matrix from Congalton 1991 Table 1.

    Used here as an example for testing. See references.txt for full citations.
    """
    mat = np.array([ [65,4,22,24],[6,81,5,8],[0,11,85,19],[4,7,3,90] ], dtype=int).view( ErrorMatrix )
    return mat

def congalton_table2():
    """
    Error matrix from Congalton 1991 Table 2.

    Used here as an example for testing. See references.txt for full citations.
    """
    sup = np.array([ [68,7,3,0],[12,112,15,10],[3,9,89,0],[0,2,5,56] ], dtype=int).view( ErrorMatrix )
    return sup

def wundram_table2():
    """
    Supervised classification results from Wundram and Loffler, 2008.

    Used as an example in Pontius and Millones 2011. See references.txt for full
    citations. Used here for testing purposes.
    """
    tab2 = np.array([   [89, 3, 7, 0, 1],
                        [11,16,10, 1, 0],
                        [ 2,14,60, 2, 0],
                        [ 1, 9, 7, 5, 0],
                        [20, 1, 2, 0, 4] ], dtype=int ).view( ErrorMatrix )
    return tab2

def wundram_table3():
    """
    Unsupervised classification results from Wundram and Loffler, 2008.

    Used as an example in Pontius and Millones 2011. See references.txt for full
    citations. Used here for testing purposes.
    """
    tab3 = np.array([   [114, 9, 8, 0, 1],
                        [  5,23,11, 1, 0],
                        [  1,10,59, 2, 0],
                        [  0, 1, 7, 5, 0],
                        [  3, 0, 1, 0, 4] ], dtype=int ).view( ErrorMatrix )
    return tab3

def ref_array():
    """
    A randomly generated array for testing.

    This was generated with: ref = np.random.randint(0,5,(10,9)). We need
    consistent results for testing so we don't regenerate it every time.
    """
    ref = np.array([   [4, 4, 3, 4, 4, 3, 2, 4, 0],
                       [4, 2, 0, 3, 0, 3, 0, 3, 3],
                       [1, 0, 3, 1, 2, 4, 0, 1, 2],
                       [0, 4, 4, 1, 3, 3, 1, 2, 0],
                       [3, 0, 1, 0, 0, 1, 3, 2, 2],
                       [4, 1, 0, 3, 4, 4, 3, 4, 3],
                       [4, 3, 4, 1, 4, 0, 0, 2, 4],
                       [0, 4, 2, 1, 1, 4, 4, 4, 4],
                       [0, 2, 1, 1, 1, 4, 0, 0, 0],
                       [4, 2, 3, 0, 4, 4, 4, 1, 0]], dtype=int)
    return ref

def comp_array():
    """
    A randomly altered version of ref_array array for testing.

    This was generated with:
    ref = ref_array()
    comp = ref.copy()
    comp = comp[np.random.rand(*comp.shape) < 0.20] = np.random.randint(0,5)
    That just replaces 20% of the numbers from ref with randomly generated
    integers.
    """
    comp = np.array([  [4, 4, 3, 4, 1, 1, 2, 4, 0],
                       [4, 1, 0, 3, 0, 3, 0, 3, 3],
                       [1, 0, 3, 1, 2, 1, 1, 1, 1],
                       [0, 4, 4, 1, 1, 3, 1, 2, 0],
                       [1, 0, 1, 1, 0, 1, 3, 2, 2],
                       [4, 1, 0, 3, 4, 4, 3, 4, 3],
                       [1, 3, 4, 1, 1, 1, 0, 2, 4],
                       [0, 4, 2, 1, 1, 4, 4, 1, 4],
                       [0, 2, 1, 1, 1, 4, 0, 0, 0],
                       [4, 2, 3, 0, 1, 4, 1, 1, 0]], dtype=int)
    return comp

#---------------------------------------------------------------#### END OF TEST DATA ####
def validate_comparison( reference, comparison ):
    """
    Take two arrays and make sure they can be compared by error_matrix.

    In order for our error matrix to make sense, the comparison array should
    not contain values that do not exist in the reference array. The numpy
    histogram2d method will bin values and return resul... oh wait, this might
    not be necessary...
    """
    return "your mom"

def error_matrix( reference, comparison, categories=None, unclassified=0 ):
    """
    Take a reference array and a comparison array and return an error matrix.

    >>> error_matrix(ref_array(),comp_array())
    ErrorMatrix([[15,  0,  0,  0],
           [ 2,  9,  0,  0],
           [ 3,  0, 13,  0],
           [ 7,  0,  0, 20]])
    """
    idx = np.where( reference<>unclassified )
    all_classes = np.unique( np.vstack( (reference[idx],comparison[idx]) ) )
    n = len( all_classes )
    em = np.array([z.count(x) for z in [zip(reference.flatten(),comparison.flatten())] for x in itertools.product(all_classes,repeat=2)]).reshape(n,n).view( ErrorMatrix )
    if categories:
        em.categories = categories
    else:
        # need vstacked values so we can check both arrays
        em.categories = all_classes.tolist()
    return em

class ErrorMatrix( np.ndarray ):
    """
    A subclass of numpy ndarray with methods to produce accuracy measures.

    >>> errmat = ErrorMatrix( np.array([ [65,4,22,24],[6,81,5,8],[0,11,85,19],[4,7,3,90] ], dtype=int) )
    >>> errmat.with_accuracies_and_totals
    ErrorMatrix([[65, 4, 22, 24, 115, 57],
           [6, 81, 5, 8, 100, 81],
           [0, 11, 85, 19, 115, 74],
           [4, 7, 3, 90, 104, 87],
           [75, 103, 115, 141, 434, None],
           [87, 79, 74, 64, None, 74]], dtype=object)


    """
    def __new__(cls, input_array, categories=None, title=None):
        """
        Constructor for new ErrorMatrix objects.

        input_array -- An N x N (square) numpy array or a string representing
        the filepath to a comma delimited csv file contaning an N x N matrix
        with just numbers (no labels or anything). There is currently no shape
        validation so non-square arrays will be accepted but most methods will
        fail or give useless results.

        categories -- A list of strings of length N containing labels for classifaction
        categories. These will be used for labeling outputs.

        title -- A string title for the error matrix that will be useful in identifying
        what classification the error matrix and accuracy measures relate to.
        """
        # Subclassing as described here: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        # Deal with a file path being handed in
        if input_array.__class__==str:
            input_array = np.genfromtxt(input_array, delimiter=',')
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        if categories:
            # if categories are given, use them
            obj.categories = categories
        else:
            # if not, just label with numbers starting at 1
            obj.categories = range(1,1+obj.shape[0])
        obj.title = title
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        if obj is None: return
        self.categories = getattr(obj, 'categories', range(1,1+self.shape[0]))
        self.title = getattr(obj, 'title', None)

    def round(self, *places):
        """
        If float type, return rounded ErrorMatrix. Otherwise just return self.
        """
        return super(ErrorMatrix, self).round( *places ).view( ErrorMatrix )

    @property
    def proportions( self ):
        """
        Return the error matrix as proportions.

        Every element of the matrix is devided by the sum of the matrix.

        >>> wundram_table2().proportions.round(4)
        ErrorMatrix([[ 0.3358,  0.0113,  0.0264,  0.    ,  0.0038],
               [ 0.0415,  0.0604,  0.0377,  0.0038,  0.    ],
               [ 0.0075,  0.0528,  0.2264,  0.0075,  0.    ],
               [ 0.0038,  0.034 ,  0.0264,  0.0189,  0.    ],
               [ 0.0755,  0.0038,  0.0075,  0.    ,  0.0151]])
        """
        return np.nan_to_num( self.astype(float) / self.sum() )

    @property
    def proportion_in_reference( self ):
        """
        Returns the proportion of pixels that fall into each category for the reference data.

        >>> wundram_table2().proportion_in_reference.round(4)
        ErrorMatrix([ 0.4642,  0.1623,  0.3245,  0.0302,  0.0189])
        """
        return np.nan_to_num( self.sum(axis=0).astype(float) / self.sum() )

    @property
    def proportion_in_comparison( self ):
        """
        Returns the proportion of pixels that fall into each category for the comparison data.

        >>> wundram_table2().proportion_in_comparison.round(4)
        ErrorMatrix([ 0.3774,  0.1434,  0.2943,  0.083 ,  0.1019])
        """
        return np.nan_to_num( self.sum(axis=1).astype(float) / self.sum() )

    def observed_proportions( self, pop_totals=None ):
        """
        Transform the matrix from observed samples to observed proportions using
        equation 1 from Pontius and Millones 2011. I'm still a little confused
        about using population totals in stratified random sampling. I need to
        look into that and do some more testing.

        >>> wundram_table2().observed_proportions().round(4)
        ErrorMatrix([[ 0.3358,  0.0113,  0.0264,  0.    ,  0.0038],
               [ 0.0415,  0.0604,  0.0377,  0.0038,  0.    ],
               [ 0.0075,  0.0528,  0.2264,  0.0075,  0.    ],
               [ 0.0038,  0.034 ,  0.0264,  0.0189,  0.    ],
               [ 0.0755,  0.0038,  0.0075,  0.    ,  0.0151]])
        """
        if pop_totals == None:
            pop_totals = self.sum(axis=1)
        return np.nan_to_num( self.astype(float) / self.sum(axis=1) ) * ( pop_totals / pop_totals.sum().astype(float) )

    @property
    def ob_props( self ):
        """
        This is just for testing at the moment so I'm not going to explain.
        """
        return self.observed_proportions()

    @property
    def commission( self ):
        """
        Returns commission disagreement values for each category.

        For a category i, commission is the total number pixels of in the comparison
        map that, according to the reference map, should have been placed in class i
        but went to other classes instead. So for category i, commission = all
        category i pixels in the comparison map - the pixels that are categorized the
        same in the reference map and the comparison map.

        The following results also confirmed by comparison with column W in the
        SampleMatrix tab of PontiusMatrix24.xlsx spreadsheet from
        http://www.clarku.edu/~rpontius after entering the correct values
        >>> wundram_table2().observed_proportions().commission.round(4)
        ErrorMatrix([ 0.0415,  0.083 ,  0.0679,  0.0642,  0.0868])
        """
        return self.sum(axis=1) - self.diagonal()

    @property
    def commission_proportion( self ):
        """
        Return error of commission as a proportion for each category.

        The following values are consistent with those published in Table 2 of
        Wundram and Loffler 2008
        >>> wundram_table2().commission_proportion.round(4)
        ErrorMatrix([ 0.11  ,  0.5789,  0.2308,  0.7727,  0.8519])
        """
        return np.nan_to_num( self.commission.astype(float) / self.sum(axis=1) )

    @property
    def omission( self ):
        """
        Returns omission disagreement values for each category.

        For a category i, omission is the total number of class i pixels from the
        reference map that ended up in non-i categories in the comparison map. So
        for category i, omission = all category i pixels in the reference map -
        the pixels that were categorized as i in both reference and comparison.

        The following results also confirmed by comparison with column X in the
        PontiusMatrix24.xlsx spreadsheet from http://www.clarku.edu/~rpontius
        >>> wundram_table2().observed_proportions().omission.round(4)
        ErrorMatrix([ 0.1283,  0.1019,  0.0981,  0.0113,  0.0038])
        """
        return self.sum(axis=0) - self.diagonal()

    @property
    def omission_proportion( self ):
        """
        Return error of omission as a proportion for each category.

        The following values are consistent with those published in Table 2 of
        Wundram and Loffler 2008 except that where I calculated a value of 0.375,
        they rounded to 0.37. I'm still pretty sure I'm doing it right.
        >>> wundram_table2().omission_proportion.round(4)
        ErrorMatrix([ 0.2764,  0.6279,  0.3023,  0.375 ,  0.2   ])
        """
        return np.nan_to_num( self.omission.astype(float) / self.sum(axis=0) )

    def quantity_disagreement_for_category( self, category ):
        """
        From equation 2 in Pontius and Millones 2011. category is an integer
        index, zero based.
        """
        return abs( self[ category ].sum() - self.T[ category ].sum() )

    @property
    def quantity_disagreements( self ):
        """
        Returns an array of quantity disagreement values. One for each category.
        """
        return abs( self.sum(axis=0) - self.sum(axis=1) ).view( np.ndarray )

    @property
    def quantity_disagreement( self ):
        """
        Returns a single quantity disagreement value for the error matrix as
        described by equation 3 in Pontius and Millones 2011.

        >>> print wundram_table2().observed_proportions().quantity_disagreement.round(4)
        0.1358
        """
        return self.quantity_disagreements.sum() / 2.0

    def allocation_disagreement_for_category( self, category ):
        """
        Returns the allocation disagreement value for a category as described
        in equation 4 of Pontuis and Millones 2011.
        """
        return 2 * np.array([ self.commission[ category ], self.omission[ category ] ]).min()

    @property
    def allocation_disagreements( self ):
        """
        Returns and array of allocation disagreement values (one for each category)
        as described by equation 4 in Pointius and Millones 2011.
        """
        return np.array( [ self.allocation_disagreement_for_category(i) for i in range( self.shape[0] ) ] )

    @property
    def allocation_disagreement( self ):
        """
        Returns a single allocation disagreement value for the whole matrix as
        described in equation 5 of Pointius and Millones 2011.

        The print statement here just keeps it from displaying as 0.20749999999999
        >>> print wundram_table2().observed_proportions().allocation_disagreement.round(4)
        0.2075
        """
        return self.allocation_disagreements.sum() / 2.0

    @property
    def overall_accuracy( self ):
        """
        Calculate total accuacy from an error matrix.

        This rounds to the same value as shown in Congalton 1991 Table 1
        >>> print congalton_table1().overall_accuracy.round(6)
        73.963134
        """
        return 100.0 * self.astype(float).diagonal().sum() / self.sum().item()

    def users_accuracy( self, category ):
        return self.users_accuracies[ category ]

    @property
    def users_accuracies( self ):
        """
        Return the user's accuracy measures for each category.

        Congalton 1991 says, 'if the total number
        of correct pixels in a category is divided by the
        total number of pixels that were classified in that
        category, then this result is a measure of commis-
        sion error. This measure, called "user's accuracy"
        or reliability, is indicative of the probability that a
        pixel classified on the map/image actually repre-
        sents that category on the ground'

        The following values match those given in Congalton 1991
        Table 1 (except these values are not rounded to whole numbers)
        >>> congalton_table1().users_accuracies.round(4)
        ErrorMatrix([ 56.5217,  81.    ,  73.913 ,  86.5385])
        """
        u_acc = 100 * self.diagonal().astype(float) / self.sum(axis=1)
        # replace nans with zeros
        if np.isnan( u_acc.sum() ):
            u_acc = np.nan_to_num( u_acc )
        return u_acc

    def producers_accuracy( self, category ):
        return self.producers_accuracies[ category ]

    @property
    def producers_accuracies( self ):
        """
        Return the producer's accuracy measures for each category.

        Congalton 1991 says, 'This accuracy measure indicates
        the probability of a reference pixel being correctly
        classified and is really a measure of omission error.'

        The following values match those given in Congalton 1991
        Table 1 (except these values are not rounded to whole numbers)
        >>> congalton_table1().producers_accuracies.round(4)
        ErrorMatrix([ 86.6667,  78.6408,  73.913 ,  63.8298])
        """
        p_acc = 100 * self.diagonal().astype(float) / self.sum(axis=0)
        # if there are nan values we want to convert to zero
        if np.isnan(p_acc.sum()):
            p_acc = np.nan_to_num( p_acc )
        return p_acc

    ##/////////////////////// ANNOTATED ARRAYS \\\\\\\\\\\\\\\\\\\\\\\\##
    def extend(self, row, col, category, corner=None):
        newem = np.vstack( (self,row) )
        if type(col) in [ ErrorMatrix, np.ndarray ]:
            col = col.tolist()
        col.append( corner )
        col = np.array([ col ]).T
        newem = np.hstack( (newem,col) ).view( ErrorMatrix )
        newem.categories = self.categories + [ category ]
        newem.title = self.title
        return newem

    def clean_zeros(self, with_unclassed=False):
        """
        Get rid of rows and columns that contain only zeros and intersect
        at the diagonal. The `with_unclassed` option lets you keep the first
        row/column pair even if it is all zeros.
        """
        emout = self.copy()
        # all zeros in row and col that meet on diagonal
        zero_col_rows = np.where((~emout.any(1)) & (~emout.any(0)))[0]
        if with_unclassed:
            # This will leave the first row and column even
            # if they do contain all zeros.
            zero_col_rows = zero_col_rows[zero_col_rows != 0]
        # print zero_col_rows
        emout = np.delete(np.delete(emout, zero_col_rows, 0), zero_col_rows, 1)
        return emout

    @property
    def with_totals( self ):
        """
        Returns an array with totals column and row added.

        >>> wundram_table2().with_totals
        ErrorMatrix([[ 89,   3,   7,   0,   1, 100],
               [ 11,  16,  10,   1,   0,  38],
               [  2,  14,  60,   2,   0,  78],
               [  1,   9,   7,   5,   0,  22],
               [ 20,   1,   2,   0,   4,  27],
               [123,  43,  86,   8,   5, 265]])
        """
        row_tots = self.sum(axis=0)
        col_tots = self.sum(axis=1)
        return self.extend( row_tots, col_tots, 'Totals', self.sum() )

    @property
    def with_accuracies( self ):
        """
        Return ErrorMatrix with accuracies added.
        """
        row = self.producers_accuracies.round().astype(int)
        col = self.users_accuracies.round().astype(int)
        corner = self.overall_accuracy.round().astype(int)
        return self.extend( row, col, 'Accuracy', corner )

    @property
    def with_accuracies_and_totals( self ):
        """
        Returns an array with colums and rows added for totals and user's
        accuracy, producer's accuracy, and overall accuracy.

        These results match those in Congalton 1991. See references.txt for
        the full citation.
        >>> congalton_table1().with_accuracies_and_totals
        ErrorMatrix([[65, 4, 22, 24, 115, 57],
               [6, 81, 5, 8, 100, 81],
               [0, 11, 85, 19, 115, 74],
               [4, 7, 3, 90, 104, 87],
               [75, 103, 115, 141, 434, None],
               [87, 79, 74, 64, None, 74]], dtype=object)


        >>> wundram_table2().with_accuracies_and_totals
        ErrorMatrix([[89, 3, 7, 0, 1, 100, 89],
               [11, 16, 10, 1, 0, 38, 42],
               [2, 14, 60, 2, 0, 78, 77],
               [1, 9, 7, 5, 0, 22, 23],
               [20, 1, 2, 0, 4, 27, 15],
               [123, 43, 86, 8, 5, 265, None],
               [72, 37, 70, 62, 80, None, 66]], dtype=object)
        """
        row = self.producers_accuracies.round().astype(int).tolist()
        col = self.users_accuracies.round().astype(int).tolist()
        row.append( None )
        col.append( None )
        corner = self.overall_accuracy.round().astype(int)
        return self.with_totals.extend( row, col, 'Accuracy', corner )

    @property
    def with_column_labels( self ):
        """
        Add column labels from self.categories.

        >>> error_matrix(ref_array(),comp_array()).with_column_labels
        ErrorMatrix([[ 1,  2,  3,  4],
               [15,  0,  0,  0],
               [ 2,  9,  0,  0],
               [ 3,  0, 13,  0],
               [ 7,  0,  0, 20]])


        >>> error_matrix(ref_array(),comp_array(),['this','that','other','thing']).with_accuracies_and_totals.with_column_labels
        ErrorMatrix([[this, that, other, thing, Totals, Accuracy],
               [15, 0, 0, 0, 15, 100],
               [2, 9, 0, 0, 11, 82],
               [3, 0, 13, 0, 16, 81],
               [7, 0, 0, 20, 27, 74],
               [27, 9, 13, 20, 69, None],
               [56, 100, 100, 100, None, 83]], dtype=object)

        """
        labels = self.categories
        cols = self.shape[1]
        if cols==len( labels ):
            newem = np.vstack( (labels, self) )
        elif cols==1 + len( labels ):
            newem = np.vstack( ( [None] + labels, self ) )
        else:
            # there was a problem probably should raise an exception
            raise Exception("Too many or too few categories for labeling while trying to label columns.")
        newem = newem.view(ErrorMatrix)
        newem.categories = self.categories
        newem.title = self.title
        return newem

    @property
    def with_row_labels( self ):
        return self.T.with_column_labels.T

    @property
    def with_labels( self ):
        """
        Add labels from self.categories to the matrix so we can export it as csv.

        >>> em = error_matrix(ref_array(),comp_array())
        >>> em.with_labels
        ErrorMatrix([[None, 1, 2, 3, 4],
               [1, 15, 0, 0, 0],
               [2, 2, 9, 0, 0],
               [3, 3, 0, 13, 0],
               [4, 7, 0, 0, 20]], dtype=object)


        >>> error_matrix(2 * ref_array(),2 * comp_array()).with_totals.with_labels
        ErrorMatrix([[None, 2, 4, 6, 8, Totals],
               [2, 15, 0, 0, 0, 15],
               [4, 2, 9, 0, 0, 11],
               [6, 3, 0, 13, 0, 16],
               [8, 7, 0, 0, 20, 27],
               [Totals, 27, 9, 13, 20, 69]], dtype=object)

        >>> error_matrix(ref_array(),comp_array(),['this','that','other','thing']).with_accuracies_and_totals.with_labels
        ErrorMatrix([[None, this, that, other, thing, Totals, Accuracy],
               [this, 15, 0, 0, 0, 15, 100],
               [that, 2, 9, 0, 0, 11, 82],
               [other, 3, 0, 13, 0, 16, 81],
               [thing, 7, 0, 0, 20, 27, 74],
               [Totals, 27, 9, 13, 20, 69, None],
               [Accuracy, 56, 100, 100, 100, None, 83]], dtype=object)
        """
        return self.with_column_labels.with_row_labels

    ###\\\\\\\\\\\\\\\\\\\\\\\\\\ END ANNOTATED ARRAYS //////////////////////

    ###--------------- OUTPUT -----------------------------------------

    def to_markdown(self):
        """
        Return a string version of the error matrix that makes a nice
        table in markdown.
        """
        emstr = self.astype('string')
        emstr[emstr == 'None'] = ''
        rows = list()
        ncols = emstr.shape[1]
        for i, row in enumerate(emstr):
            rows.append('|' + '|'.join(row) + '|')
            if i == 0:
                rows.append('-'.join(np.repeat('|', ncols + 1)))
        return '\n'.join(rows)

    def to_dataframe(self, cmap=None):
        """
        Return a pandas dataframe version of the `ErrorMatrix`. Do not
        use with the `with_labels` property. Labels are included without that
        and it will make this function fail.

        Parameters
        ----------
        cmap : matplotlib colormap or `True`
            If `None` (default), the returned dataframe will not be styled with
            background colors. Otherwise cell colors will be added to the error
            matrix when the data frame is viewed in Jupyter Notebook (aka
            IPython Notebook). If `True` one of two default colormaps will be
            used. First, an attempt will be made to get a colormap from seaborn.
            If seaborn is not installed, an attempt will be made to get a
            matplotlib colormap (that's a bit uglier). The use can also supply
            their own colormap instead.

        Returns
        -------
        pandas dataframe or dataframe styler
            A dataframe representation of the error matrix that looks nice in a
            Jupyter Notebook. If a cmap is applied, a `pandas.core.style.Styler`
            object will be returned. The dataframe can be accessed via the
            `.data` property of the `Styler`.
        """
        import pandas as pd
        df = pd.DataFrame(self, columns=self.categories, index=self.categories)
        df = df.replace('None',np.nan)
        if cmap is None:
            return df
        else:
            if cmap is True:
                # Try to provide a default color map
                try:
                    from seaborn import light_palette
                    cmap = light_palette('steelblue', as_cmap=True)
                except ImportError:
                    # seaborn is less common than matplotlib. I don't really
                    # want to make either one a dependency for this module.
                    import matplotlib.pyplot as plt
                    cmap = plt.cm.GnBu
            subst = df.columns.difference(['Totals','Accuracy'])
            return df.style.background_gradient(cmap=cmap,
                                                subset=(subst, subst))

    def to_latex(self, **kwargs):
        ff = kwargs.pop('float_format', '%.f')
        nr = kwargs.pop('na_rep', '-')
        return self.to_dataframe().to_latex(float_format=ff,
                                            na_rep=nr, **kwargs)

    def save_csv( self, filepath, annotations=['with_accuracies_and_totals','with_labels'], rounding=None ):
        with open(filepath, 'wb') as f:
            writer = csv.writer(f)
            # copy the array so we can annotate it without screwing it up
            arr = self.copy()
            # round if needed
            if rounding:
                arr = arr.round( rounding )
            # annotate as specified
            for a in annotations: arr = arr.__getattribute__( a )
            # write the title if there is one
            if self.title:
                writer.writerow( [ str( self.title ) ] )
            # write the annotated error matrix
            writer.writerows( arr )
            # write an empty row as a spacer
            writer.writerow( [''] )
            # write quantity and allocation disagreements
            writer.writerow( ['Quantity Disagreement',self.quantity_disagreement] )
            writer.writerow( ['Allocation Disagreement',self.allocation_disagreement] )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
