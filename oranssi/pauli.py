from typing import List, Union
import copy
import numpy as np

### Code to rewrite pauli strings ###

levicevita = lambda i, j, k: (i - j) * (j - k) * (k - i) / 2
pauli_idx_str2int_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
pauli_idx_int2str_map = dict(zip(pauli_idx_str2int_map.values(),
                                 pauli_idx_str2int_map.keys()))


class Pauli(object):
    def __init__(self, alpha: int, gamma: Union[int, str], coeff: Union[float, complex] = 1.0):
        assert isinstance(alpha, (int, np.int)), f'alpha must be `int`, received `{type(alpha)}`'
        assert alpha >= 0, f'alpha must be >= 0, found {alpha}'
        self.alpha = alpha
        assert isinstance(gamma, (int, np.integer, str)), f'gamma must be `int` or `str`, received `{type(gamma)}`'
        assert isinstance(coeff, (
            float, np.float, complex, np.complex)), f'coeff must be `complex` or `float`, received `{type(gamma)}`'
        # Convert internally to integer indices (0,x,y,z)->(0,1,2,3)
        if isinstance(gamma, str):
            assert gamma in pauli_idx_str2int_map.keys(), 'gamma index must be ' \
                                                          f'`{sorted(list(pauli_idx_str2int_map.keys()))}`, ' \
                                                          f'received `{gamma}`'
            self.gamma = pauli_idx_str2int_map[gamma]
        else:
            assert gamma in pauli_idx_str2int_map.values(), 'gamma index must be ' \
                                                            f'`{sorted(list(pauli_idx_str2int_map.values()))}`, ' \
                                                            'received `{gamma}`'
            self.gamma = gamma
        # TODO: add support for symbolic coefficients here. Maybe with sympy?
        # For now we cast to numpy for consistency down the road
        self.coeff = np.complex(coeff)

    def __mul__(self, other):
        # Multiplication means applying the SU(2) commutator [s^a, s^b] = -i lvc_abc s^c + delta_{ab}
        assert self.alpha == other.alpha, f'Sites must be equal for multiplication, ' \
                                          f'found {self.alpha} and {other.alpha}'
        # Case where I=I, X=X, Y=Y, Z=Z
        if self.gamma == other.gamma:
            return Pauli(self.alpha, 0, coeff=self.coeff * other.coeff)
        # Pauli 1 is I
        elif self.gamma == 0:
            return Pauli(self.alpha, other.gamma, coeff=self.coeff * other.coeff)
        # Pauli 2 is I
        elif other.gamma == 0:
            return Pauli(self.alpha, self.gamma, coeff=self.coeff * other.coeff)
        # Pauli 1 and 2 are x,y,z but not equal, apply levi cevita symbol
        else:
            new_gamma = [x for x in [1, 2, 3] if x not in [self.gamma, other.gamma]][0]
            return Pauli(self.alpha, new_gamma,
                         coeff=1j * levicevita(self.gamma, other.gamma, new_gamma) * self.coeff * other.coeff)

    def __eq__(self, other):
        # Paulis are equal if their indices are the same and they exist on the same site.
        # Note that we do not check if the coefficients are equal!
        return ((self.alpha == other.alpha) & (self.gamma == other.gamma))

    def __tex__(self, coeff: bool = True):
        # Convert to readable tex
        if coeff:
            if self.gamma == 0:
                return '{} I'.format(self.coeff)
            else:
                return '{} \sigma_{{{}}}^{{{}}}'.format(self.coeff, self.alpha, pauli_idx_int2str_map[self.gamma])
        else:
            if self.gamma == 0:
                return 'I_{{{}}}'.format(self.alpha)
            else:
                return '\sigma_{{{}}}^{{{}}}'.format(self.alpha, pauli_idx_int2str_map[self.gamma])

    def __repr__(self, coeff: bool = True):
        # unique representation of the pauli, also used for hashing
        if coeff:
            if self.coeff == 1.0:
                return f'S({self.alpha},{self.gamma})'
            else:
                return f'{self.coeff} * S({self.alpha},{self.gamma})'
        else:
            return f'S({self.alpha},{self.gamma})'

    def __hash__(self):
        # unique hash representation of the object
        return hash(repr(self))

    def tex(self):
        # method access to tex
        return self.__tex__()

    def conj(self):
        # return the conjugate of the pauli by taking he complex conjugate of the coefficient
        self.coeff = np.conj(self.coeff)
        return self


class PauliMonomial(object):
    def __init__(self, list_of_paulis: List[Pauli]):
        assert all(isinstance(p, Pauli) for p in
                   list_of_paulis), \
            'All objects in list_of_paulis must be `Pauli` objects, found ' \
            f'{list(filter(lambda x: True if not isinstance(x, Pauli) else False, list_of_paulis))} ' \
            'in list_of_paulis with respective type ' \
            f'{[type(y) for y in filter(lambda x: not isinstance(x, Pauli), list_of_paulis)]}'
        # we sort the paulis based on the site
        list_of_paulis = sorted(list_of_paulis, key=lambda x: x.alpha)
        # we store the locations (useful later for multiplications)
        locations = [p.alpha for p in list_of_paulis]
        assert len(set(locations)) == len(list_of_paulis), \
            'All locations of the Pauli objects in list_of_paulis must be unique, found multiple occurences: ' \
            f'{[(loc, locations.count(loc)) for loc in set(locations) if locations.count(loc) > 1]} ' \
            f'where the tuple indicates (location, number of occurences)'
        self.locations = locations
        self.monomial = list_of_paulis
        # get the total coefficient from all the individual coefficients
        self.coeff = np.prod([p.coeff for p in self.monomial])

    def __getitem__(self, item):
        # this makes it possible to call the Paulimonomial as paulimon[i]
        return self.monomial[item]

    def __mul__(self, other):
        # enable multiplication of monomials
        new_monomial = []
        # Create sets for O(1) lookup
        set_self_locations = set(self.locations)
        set_other_locations = set(other.locations)
        # Get the sites that both monomials have in common and the ones that are unique.
        sites_in_common = set_self_locations & set_other_locations
        sites_self_unique = set_self_locations - sites_in_common
        sites_other_unique = set_other_locations - sites_in_common
        # Take the product of the common paulis and add them
        for i in sites_in_common:
            new_monomial.append(self[self.locations.index(i)] * other[other.locations.index(i)])
        # Create new Pauli objects for the unique locations so we don't get reference issues
        for j in sites_self_unique:
            p = self[self.locations.index(j)]
            new_monomial.append(Pauli(p.alpha, p.gamma, p.coeff))
        for k in sites_other_unique:
            p = other[other.locations.index(k)]
            new_monomial.append(Pauli(p.alpha, p.gamma, p.coeff))
        # creating a new PauliMonomial will automatically sort based on site
        return PauliMonomial(new_monomial)

    def __eq__(self, other):
        # monomials are equal of all paulis are equal. If they are not of equal length they cannot be compared and
        # we raise an error
        if len(self.monomial) != len(other.monomial):
            raise AssertionError(
                f'Can only evaluate `==` if both monomials are of equal length, '
                f'found {len(self.monomial)} and {len(other.monomial)}')
        if ((np.isreal(self.coeff) & np.isreal(other.coeff)) | (
                ~np.isreal(self.coeff) & ~np.isreal(other.coeff))):
            return all([p == q for p, q in zip(self.monomial, other.monomial)])
        else:
            return False

    def __len__(self):
        return len(self.monomial)

    def __lt__(self, other):
        # In order to utilize vectorized numpy methods for checking which elements of an array of PauliMonomials is
        # equal, we have to implement a `<` operator so that the array can be sorted. We require here an __lt__ method
        # that satisfies the following constraints: PauliMonomials must be first grouped by upper index gamma=1,x,y,z,
        # then by whether or not the coefficients of the monomials are complex, and finaly by sign of the coefficients.
        # the following approach seems to satisfy those constraints:

        # First check if the gammas are less or equal in order of appearance in the pauli monomial
        self_gammas = [m.gamma for m in self.monomial]
        other_gammas = [m.gamma for m in other.monomial]
        if self_gammas == other_gammas:
            # if they are equal we decide on wether the coefficients are smaller
            # we always say that a real coefficient is larger than a complex one
            if (~np.isreal(self.coeff) & np.isreal(other.coeff)):
                return True
            elif (np.isreal(self.coeff) & ~np.isreal(other.coeff)):
                return True
            # if the coefficients are in the same domain, we simply decide by sign.
            else:
                # can now be either +-a + i0 < +-a' + i0 or 0 + +-ib < 0 + +-ib'
                return ((np.real(self.coeff) + np.imag(self.coeff)) < (np.real(other.coeff) + np.imag(other.coeff)))
        # if the monomials are not equal we can just decide based on the gamma indices of the paulis (I<x<y<z)
        else:
            return self_gammas < other_gammas

    def __tex__(self, coeff=True):
        # convert to tex
        if coeff:
            return f'{self.coeff} ' + ' '.join(p.__tex__(coeff=False) for p in self.monomial)
        else:
            return f' '.join(p.__tex__(coeff=coeff) for p in self.monomial)

    def __repr__(self, coeff=True):
        # create readable string representation
        if coeff:
            if self.coeff == 1.0:
                return ' '.join(p.__repr__(coeff=False) for p in self.monomial)
            else:
                return f'{self.coeff} * ' + ' '.join(p.__repr__(coeff=False) for p in self.monomial)
        else:
            return ' '.join(p.__repr__(coeff=False) for p in self.monomial)

    def __hash__(self):
        # create unique has for set usage
        return hash(
            f'{"re" if np.isreal(self.coeff) else "im"}' + ' '.join(p.__repr__(coeff=False) for p in self.monomial))

    def tex(self, coeff=True):
        # method for tex
        if coeff:
            return self.__tex__()
        else:
            return self.__tex__(coeff=False)

    def conj(self):
        # conjugate all paulis in the monomial
        for p in self.monomial:
            p.conj()
        return self

    def commutes_with(self, operator):
        # if A and B are monomials, AB and BA will be equal up to their coefficient. Since or equality method does not
        # check if the coefficients are equal, we do it there.
        assert isinstance(operator,
                          type(self)), f'expected `operator` to be a PauliMonomial object, received {type(operator)}'
        # A*B
        left_side = self * operator
        # B*A
        right_side = operator * self
        return (right_side.coeff == left_side.coeff)

    def commutator(self, operator):
        # return AB-BA
        assert isinstance(operator,
                          type(self)), f'expected `operator` to be a PauliMonomial object, received {type(operator)}'
        # A*B
        left_side = self * operator
        # B*A
        right_side = operator * self
        left_side.coeff = left_side.coeff - right_side.coeff
        return left_side


class PauliSum(object):
    def __init__(self, list_of_pauli_monomials: List[PauliMonomial]):
        assert all(isinstance(p, PauliMonomial) for p in list_of_pauli_monomials), \
            'All objects in list_of_paulis must be `PauliMonomial` objects, found ' \
            f'{list(filter(lambda x: True if not isinstance(x, PauliMonomial) else False, list_of_pauli_monomials))} ' \
            'in list_of_paulis with respective type ' \
            f'{[type(y) for y in filter(lambda x: not isinstance(x, Pauli), list_of_pauli_monomials)]}'
        # make sure all monomials are of equal length
        assert all(len(ps) == len(list_of_pauli_monomials[0]) for ps in list_of_pauli_monomials), \
            'All PauliMonomials must have the same length, found multiple different length: ' \
            f'{[len(ps) for ps in list_of_pauli_monomials]} ' \
            f'where the tuple indicates (location, number of occurences)'
        self.pauli_sum = list_of_pauli_monomials
        self.simplify()

    def __getitem__(self, item):
        # this makes it possible to call the Paulimonomial as paulimon[i]
        return self.pauli_sum[item]

    def simplify(self):
        duplicates = set([i for i in self.pauli_sum if self.pauli_sum.count(i)>1])
        if not len(duplicates):
            return
        else:
            # Loop over the duplicates
            for dup in duplicates:
                coeff = 0
                remove_idx = []
                # collect total coefficient and store indices of duplicates in pauli_sum
                for j, ps in enumerate(self.pauli_sum):
                    if ps == dup:
                        coeff += ps.coeff
                        remove_idx.append(j)
                if len(remove_idx)>1:
                    self.pauli_sum = [i for j, i in enumerate(self.pauli_sum) if j not in remove_idx[1:]]
                # combine all coefficients into the first occurence
                self.pauli_sum[remove_idx[0]].coeff = coeff
        self.pauli_sum = [ps for ps in self.pauli_sum if ~np.isclose(ps.coeff,0.0)]

    def __len__(self):
        return len(self.pauli_sum)
    def __mul__(self, other):
        # enable multiplication of monomials
        new_pauli_sum = []
        for ps_i in self.pauli_sum:
            for ps_j in other.pauli_sum:
                new_pauli_sum.append(ps_i*ps_j)
        return PauliSum(new_pauli_sum)

    def __add__(self, other):
        # enable multiplication of monomials
        new_pauli_sum = copy.copy(self.pauli_sum) + copy.copy(other.pauli_sum)
        return PauliSum(new_pauli_sum)

    def __sub__(self, other):
        # enable multiplication of monomials
        other_pauli_sum = copy.copy(other)
        other_pauli_sum.scalar_mul(-1)
        new_pauli_sum = copy.copy(self.pauli_sum) + other_pauli_sum.pauli_sum
        return PauliSum(new_pauli_sum)

    def scalar_mul(self, scalar:Union[int,float, complex]):
        assert isinstance(scalar, (int, float, complex)), f'`scalar` must be a float, received type(scalar) = {type(scalar)}'
        for ps in self.pauli_sum:
            ps.coeff*=scalar