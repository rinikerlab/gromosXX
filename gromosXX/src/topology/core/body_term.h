/**
 * @file body_term.h
 * define the structures to hold topology information
 * of the n-body terms (like bonds, angles, dihedrals).
 */

#ifndef INCLUDED_BODY_TERM_H
#define INCLUDED_BODY_TERM_H

namespace topology {

  /**
   * @enum functional_form
   * functional form
   */
  enum functional_form {
    /**
     * full harmonic (attractive and repulsive)
     */
    harmonic = 0,
    /**
     * half harmonic, attractive
     */
    attractive = 1,
    /**
     * half harmonic, repulsive
     */
    repulsive = -1
  };

  /**
   * @struct one_body_term_struct
   * one body terms (topological information).
   */
  struct one_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param t interaction type.
     */
    one_body_term_struct(unsigned int i, unsigned int t) : i(i), type(t) {
    };

    /**
     * atom i.
     */
    unsigned int i;
    /**
     * interaction type.
     */
    unsigned int type;

    /**
     * equal operator
     */
    bool operator==(one_body_term_struct const & b) {
      return (b.i == i && b.type == type);
    }

  };

  /**
   * @struct perturbed_one_body_term_struct
   * perturbed one body terms (topological information).
   */
  struct perturbed_one_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param t_A interaction type.
     * @param t_B interaction type.
     */
    perturbed_one_body_term_struct(unsigned int i, unsigned int t_A, unsigned int t_B)
    : i(i), A_type(t_A), B_type(t_B) {
    };

    /**
     * atom i.
     */
    unsigned int i;
    /**
     * interaction type for state A.
     */
    unsigned int A_type;

    /**
     * interaction type for state B.
     */
    unsigned int B_type;

    /**
     * equal operator
     */
    bool operator==(perturbed_one_body_term_struct const & b) {
      return (b.i == i && b.A_type == A_type && b.B_type == B_type);
    }

  };

  /**
   * @struct two_body_term_struct
   * two body term interaction topological information.
   */
  struct two_body_term_struct : public one_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param j atom j.
     * @param t interaction type.
     */
    two_body_term_struct(unsigned int i, unsigned int j, unsigned int t)
    : one_body_term_struct(i, t), j(j) {
    };

    /**
     * atom j.
     */
    unsigned int j;

    /**
     * equal operator
     */
    bool operator==(two_body_term_struct const & b) {
      return (one_body_term_struct::operator==(b) && b.j == j);
    }

  };

  /**
   * @struct perturbed_two_body_term_struct
   * perturbed two body term interaction topological information.
   */
  struct perturbed_two_body_term_struct : public perturbed_one_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param j atom j.
     * @param t_A interaction type for state A.
     * @param t_B interaction type for state B.
     */
    perturbed_two_body_term_struct(unsigned int i, unsigned int j, unsigned int t_A, unsigned int t_B)
    : perturbed_one_body_term_struct(i, t_A, t_B), j(j) {
    };

    /**
     * atom j.
     */
    unsigned int j;

    /**
     * equal operator
     */
    bool operator==(perturbed_two_body_term_struct const & b) {
      return (perturbed_one_body_term_struct::operator==(b) && b.j == j);
    }
  };

  /**
   * @struct three_body_term_struct
   * three body term interaction topological information.
   */
  struct three_body_term_struct : public two_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param j atom j.
     * @param k atom k.
     * @param t interaction type.
     */
    three_body_term_struct(unsigned int i, unsigned int j, unsigned int k, unsigned int t)
    : two_body_term_struct(i, j, t), k(k) {
    };

    /**
     * atom k.
     */
    unsigned int k;

    /**
     * equal operator
     */
    bool operator==(three_body_term_struct const & b) {
      return (two_body_term_struct::operator==(b) && b.k == k);
    }

  };

  /**
   * @struct perturbed_three_body_term_struct
   * perturbed three body term interaction topological information.
   */
  struct perturbed_three_body_term_struct : public perturbed_two_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param j atom j.
     * @param k atom k.
     * @param t_A interaction type for state A.
     * @param t_B interaction type for state B.
     */
    perturbed_three_body_term_struct(unsigned int i, unsigned int j, unsigned int k,
            unsigned int t_A, unsigned int t_B)
    : perturbed_two_body_term_struct(i, j, t_A, t_B), k(k) {
    };

    /**
     * atom k.
     */
    unsigned int k;

    /**
     * equal operator
     */
    bool operator==(perturbed_three_body_term_struct const & b) {
      return (perturbed_two_body_term_struct::operator==(b) && b.k == k);
    }
  };

  /**
   * @struct four_body_term_struct
   * four body term interaction topological information.
   */
  struct four_body_term_struct : public three_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param j atom j.
     * @param k atom k.
     * @param l atom l.
     * @param t interaction type.
     */
    four_body_term_struct(unsigned int i, unsigned int j, unsigned int k, unsigned int l, unsigned int t)
    : three_body_term_struct(i, j, k, t), l(l) {
    };

    /**
     * atom l.
     */
    unsigned int l;

    /**
     * equal operator
     */
    bool operator==(four_body_term_struct const & b) {
      return (three_body_term_struct::operator==(b) && b.l == l);
    }

  };

  /**
   * @struct perturbed_four_body_term_struct
   * perturbed four body term interaction topological information.
   */
  struct perturbed_four_body_term_struct : public perturbed_three_body_term_struct {

    /**
     * Constructor.
     * @param i atom i.
     * @param j atom j.
     * @param k atom k.
     * @param l atom l.
     * @param t_A interaction type for state A.
     * @param t_B interaction type for state B.
     */
    perturbed_four_body_term_struct(unsigned int i, unsigned int j, unsigned int k, unsigned int l,
            unsigned int t_A, unsigned int t_B)
    : perturbed_three_body_term_struct(i, j, k, t_A, t_B), l(l) {
    };

    /**
     * atom l.
     */
    unsigned int l;

    /**
     * equal operator
     */
    bool operator==(perturbed_four_body_term_struct const & b) {
      return (perturbed_three_body_term_struct::operator==(b) && b.l == l);
    }
  };

  /**
   * @struct eight_body_term_struct
   * eight body terms (topological information).
   */
  struct eight_body_term_struct {

    /**
     * Constructor.
     * @param a atom a.
     * @param b atom b.
     * @param c atom c.
     * @param d atom d.
     * @param e atom g.
     * @param f atom e.
     * @param g atom f.
     * @param h atom h.
     * @param t interaction type.
     */
    eight_body_term_struct(unsigned int a, unsigned int b, unsigned int c,
                           unsigned int d, unsigned int e, unsigned int f,
                           unsigned int g, unsigned int h, unsigned int t) :
                           a(a), b(b), c(c), d(d), e(e), f(f), g(g), h(h), type(t) {
    };

    /**
     * atom a.
     */
    unsigned int a;
    /**
     * atom a.
     */
    unsigned int b;
    /**
     * atom a.
     */
    unsigned int c;
    /**
     * atom a.
     */
    unsigned int d;
    /**
     * atom a.
     */
    unsigned int e;
    /**
     * atom a.
     */
    unsigned int f;
    /**
     * atom a.
     */
    unsigned int g;
    /**
     * atom a.
     */
    unsigned int h;
    /**
     * interaction type.
     */
    unsigned int type;

    /**
     * equal operator
     */
    bool operator==(eight_body_term_struct const & i) {
      return (i.a == a && i.b == b && i.c == c && i.d == d && i.e == e &&
              i.f == f && i.g == g && i.h == h && i.type == type);
    }

  };

  /**
   * @struct perturbed_eight_body_term_struct
   * perturbed eight body terms (topological information).
   */
  struct perturbed_eight_body_term_struct {

    /**
     * Constructor.
     * @param a atom a.
     * @param b atom b.
     * @param c atom c.
     * @param d atom d.
     * @param e atom g.
     * @param f atom e.
     * @param g atom f.
     * @param h atom h.
     * @param t_A interaction A_type.
     * @param t_B interaction B_type.
     */
    perturbed_eight_body_term_struct(unsigned int a, unsigned int b,
                                     unsigned int c, unsigned int d,
                                     unsigned int e, unsigned int f,
                                     unsigned int g, unsigned int h,
                                     unsigned int t_A, unsigned int t_B) :
                                     a(a), b(b), c(c), d(d), e(e), f(f),
                                     g(g), h(h), A_type(t_A), B_type(t_B) {
    };

    /**
     * atom a.
     */
    unsigned int a;
    /**
     * atom a.
     */
    unsigned int b;
    /**
     * atom a.
     */
    unsigned int c;
    /**
     * atom a.
     */
    unsigned int d;
    /**
     * atom a.
     */
    unsigned int e;
    /**
     * atom a.
     */
    unsigned int f;
    /**
     * atom a.
     */
    unsigned int g;
    /**
     * atom a.
     */
    unsigned int h;
    /**
     * interaction type state A.
     */
    unsigned int A_type;
    /**
     * interaction type state B.
     */
    unsigned int B_type;
    /**
     * equal operator
     */
    bool operator==(perturbed_eight_body_term_struct const & i) {
      return (i.a == a && i.b == b && i.c == c && i.d == d && i.e == e &&
              i.f == f && i.g == g && i.h == h && i.A_type == A_type
              && i.B_type == B_type);
    }

  };

  /**
   * Position restraints information.
   */
  struct position_restraint_struct {

    /**
     * Constructor.
     */
    position_restraint_struct(unsigned int seq)
    : seq(seq) {
    }
    /**
     * sequence number.
     */
    unsigned int seq;
  };

  struct distance_restraint_struct {

    /**
     * Constructor.
     */
    distance_restraint_struct(util::Virtual_Atom v1,
            util::Virtual_Atom v2,
            double r0, double w0, int rah)
    : v1(v1), v2(v2),
    r0(r0),
    w0(w0),
    rah(rah) {
    }

    /**
     * Virtual Atom 1.
     */
    util::Virtual_Atom v1;

    /**
     * Virtual Atom 2.
     */
    util::Virtual_Atom v2;

    /**
     * restraint distance.
     */
    double r0;
    /**
     * weighting factor.
     */
    double w0;
    /**
     *repulsiv, attractiv, harmonic
     */
    int rah;

  };

  /**
   *Perturbed distance restraints information.
   */
  struct perturbed_distance_restraint_struct {

    /**
     * Constructor.
     */
    perturbed_distance_restraint_struct(util::Virtual_Atom v1,
            util::Virtual_Atom v2,
            int n, int m,
            double A_r0, double B_r0,
            double A_w0, double B_w0, int rah)
    : v1(v1), v2(v2),
    n(n), m(m),
    A_r0(A_r0), B_r0(B_r0),
    A_w0(A_w0), B_w0(B_w0),
    rah(rah) {
    }

    /**
     * Virtual Atom 1.
     */
    util::Virtual_Atom v1;

    /**
     * Virtual Atom 2.
     */
    util::Virtual_Atom v2;

    /**
     * hidden restraint factor n
     */
    int n;

    /**
     * hidden restraint factor m
     */
    int m;

    /**
     * restraint distance A.
     */
    double A_r0;
    /**
     * restraint distance B.
     */
    double B_r0;
    /**
     * weighting factor A.
     */
    double A_w0;
    /**
     * weighting factor B.
     */
    double B_w0;
    /**
     *repulsiv, attractiv, harmonic
     */
    int rah;
  };

  /**
   * eds distance restraints information.
   */
  struct eds_distance_restraint_struct {

    /**
     * Constructor.
     */
    eds_distance_restraint_struct(util::Virtual_Atom v1,
            util::Virtual_Atom v2,
            std::vector<double> r0, std::vector<double> w0, int rah)
    : v1(v1), v2(v2),
    r0(r0),
    w0(w0),
    rah(rah) {
    }

    /**
     * Virtual Atom 1.
     */
    util::Virtual_Atom v1;

    /**
     * Virtual Atom 2.
     */
    util::Virtual_Atom v2;

    /**
     * restraint distance in the different states.
     */
    std::vector<double> r0;
    /**
     * weighting factor in the different states.
     */
    std::vector<double> w0;
    /**
     *repulsive, attractiv, harmonic
     */
    int rah;

  };

  struct dihedral_restraint_struct {

    /**
     * Constructor.
     */
    dihedral_restraint_struct(int i, int j, int k, int l, double delta, double phi, double w0)
    : i(i), j(j), k(k), l(l),
    delta(delta), phi(phi),
    w0(w0) {
    }

    /**
     * atom i
     */
    int i;
    /**
     * atom j
     */
    int j;
    /**
     * atom k
     */
    int k;
    /**
     * atom l
     */
    int l;

    /**
     * restraint maximum
     * (periodicity shift)
     */
    double delta;
    /**
     * restraint angle
     */
    double phi;
    /**
     * weighting factor.
     */
    double w0;
  };

  struct perturbed_dihedral_restraint_struct {

    /**
     * Constructor.
     */
    perturbed_dihedral_restraint_struct(int i, int j, int k, int l, int m, int n, double delta,
            double A_phi, double A_w0, double B_phi, double B_w0)
    : i(i), j(j), k(k), l(l),
    m(m), n(n),
    delta(delta),
    A_phi(A_phi), A_w0(A_w0),
    B_phi(B_phi), B_w0(B_w0) {
    }

    /**
     * atom i
     */
    int i;
    /**
     * atom j
     */
    int j;
    /**
     * atom k
     */
    int k;
    /**
     * atom l
     */
    int l;
    /**
     * exponent m
     */
    int m;
    /**
     * exponent n
     */
    int n;
    /**
     * restraint maximum
     * (periodicity shift)
     */
    double delta;
    /**
     * restraint angle state A
     */
    double A_phi;
    /**
     * weighting factor state A
     */
    double A_w0;
    /**
     * restraint angle state B
     */
    double B_phi;
    /**
     * weighting factor state B
     */
    double B_w0;
  };

  /**
   * Virtual Grain
   */
  struct virtual_grain_struct {

    /**
     * Constructor
     */
    virtual_grain_struct(int i, util::Virtual_Atom va)
    : i(i), atom(va) {
    }
    /**
     * virtual atom index
     */
    int i;
    /**
     * virtual atom
     */
    util::Virtual_Atom atom;
  };

  /**
   * J-Value restraints.
   */
  struct jvalue_restraint_struct {

    /**
     * Constructor.
     */
    jvalue_restraint_struct(int i, int j, int k, int l,
            double K, double J0,
            double a, double b, double c,
            double delta,
            functional_form H)
    : i(i), j(j), k(k), l(l),
    K(K), J0(J0),
    a(a), b(b), c(c), delta(delta),
    H(H) {
    }

    /**
     * atom sequence numbers.
     */
    unsigned int i, j, k, l;
    /**
     * force constant
     */
    double K;
    /**
     * J0
     */
    double J0;
    /**
     * Karplus parameter.
     */
    double a, b, c;
    /**
     * phase shift
     */
    double delta;
    /**
     * functional form.
     * - half harmonic attractive,
     * - half harmonic repulsive,
     * - harmonic
     */
    functional_form H;
  };

  /**
   * Xray restraints.
   */
  struct xray_restraint_struct {

    /**
     * Constructor.
     */
    xray_restraint_struct(int h, int k, int l,
            double sf, double stddevsf)
    : h(h), k(k), l(l),
    sf(sf), stddev_sf(stddev_sf) {
    }

    /**
     * grid indices.
     */
    int h, k, l;
    /**
     * structure factor
     */
    double sf;
    /**
     * standard-deviation of structure factor
     */
    double stddev_sf;
  };

  /**
   * xray umbrella weights
   */
  struct xray_umbrella_weight_struct {

    /**
     * constructor
     */
    xray_umbrella_weight_struct(int id, double threshold, double threshold_growth_rate, bool threshold_freeze, double cutoff, std::vector<unsigned int> atoms)
    : id(id), threshold(threshold), threshold_growth_rate(threshold_growth_rate), threshold_freeze(threshold_freeze), cutoff(cutoff), atoms(atoms) {
    }
    /**
     * the ID of the umbrella
     */
    int id;
    /**
     * the threshold
     */
    double threshold;
    /**
     * is the threshold still growing?
     */
    double threshold_growth_rate;
    /**
     * signal grow, freeze threshold
     */
    bool threshold_freeze;
    /**
     * the cutoff
     */
    double cutoff;
    /**
     * the atoms attched to the umbrella weigh
     */
    std::vector<unsigned int> atoms;
  };

}

#endif
