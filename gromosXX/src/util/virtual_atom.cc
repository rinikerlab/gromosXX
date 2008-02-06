/**
 * @file virtual_atom.cc
 * virtual atoms
 */

#include <stdheader.h>
#include <util/debug.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>

#include "virtual_atom.h"

#include <math/periodicity.h>
#include <util/template_split.h>

#undef MODULE
#undef SUBMODULE
#define MODULE util
#define SUBMODULE util

static const double TETHCO=0.577350;
static const double TETHSI=0.816497;




util::Virtual_Atom::Virtual_Atom()
  :  m_type(va_explicit),
     m_atom(),
     m_dish(0.1),
     m_disc(0.153),
     m_orientation(0)
{
}

util::Virtual_Atom::Virtual_Atom(virtual_type type, std::vector<int> atom,
				 double dish, double disc,int orientation)
  :  m_type(type),
     m_atom(atom),
     m_dish(dish),
     m_disc(disc),
     m_orientation(orientation)
{
  bool strict = true; // do the test?
  unsigned int expected = 0; // number of atoms expected for virtual atom
                             // type
  switch(m_type) {
    case 0: // explicit atom
    case 7: // rotating ring
      expected = 1;
      break;
    case 1: // CH1
      expected = 4;
      break;
    case 2: // aromatic H
    case 3: // non-stereospecific CH2
    case 4: // stereospecific CH2
    case 6: // non-stereospecific CH3 (Leu, Val)
      expected = 3;
      break;
    case 5: // CH3
    case 8: // NH2-group (one pseudosite)
    case 9: // (CH3)3-group (one psuedosite)
      expected = 2;
      break;
    case 10: // cog
      strict = false;
      break;
    default:
      io::messages.add("Virual Atom", "wrong type", io::message::error);
  }

  if (strict && expected != m_atom.size()) {
    std::ostringstream oss;
    oss << "VA ";
    std::vector<int>::const_iterator it = m_atom.begin(), to = m_atom.end();
    for(; it != to; ++it)
      oss << *it << ' ';
    oss << "of type " << m_type << " has " << m_atom.size() << " atom(s) but "
           "expected " << expected << " atom(s)";
    io::messages.add("Virtual Atom", oss.str(), io::message::error);
  }
}

template<math::boundary_enum B>
void util::Virtual_Atom::_pos
(
 math::VArray const & position,
 math::Box const & box,
 math::Vec & p
)const
{
  math::Vec s,t, posi, posj, posk, posl;
  math::Periodicity<B> periodicity(box);

  posi = position(m_atom[0]);

  switch(m_type){
    
    case 0: // explicit atom
    case 7: // rotating ring
      assert(m_atom.size()>0);
      p = posi; 
      break;
      
    case 1: // CH1
    
      assert(m_atom.size()>3);

      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;
      periodicity.nearest_image(position(m_atom[3]), posi, posl);
      posl += posi;
      
      s = 3.0 * posi - posj - posk - posl;
      p = posi + m_dish / math::abs(s) * s;
      break;

    case 2: // aromatic H
      assert(m_atom.size()>2);

      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;
       
      s = 2.0 * posi - posj - posk;
      p = posi + m_dish / math::abs(s) * s;
      break;
      
    case 3: // non-stereospecific CH2
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;

      s = 2.0 * posi - posj - posk;
      p = posi + m_dish * TETHCO / math::abs(s) * s;
      break;
      
    case 4: // stereospecific CH2
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;
      
      s = 2.0 * posi - posj - posk;
      DEBUG(10, "\ts = " << math::v2s(s));
      
      t = math::cross(posi - posj, posi - posk);
      DEBUG(10, "\tq = " << math::v2s(t));
      
      DEBUG(10, "\tDISH = " << m_dish << "\tTETHCO = " << TETHCO << "\tTETHSI = " << TETHSI);
      p =  posi + m_dish * TETHCO / math::abs(s) * s + m_dish * TETHSI / math::abs(t) * t;
      break;
      
    case 5: // CH3
      assert(m_atom.size()>1);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      
      s =  posi - posj;
      p = posi + m_dish / (3 * math::abs(s)) * s;
      break;
      
    case 6: // non-stereospecific CH3 (Leu, Val)
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;
      
      s = 2.0 * posi - posj - posk;
      p = posi - TETHCO * (m_disc + m_dish / 3.0) / math::abs(s) * s;
      break;
      
    case 8: // NH2-group (one pseudosite)
      assert(m_atom.size()>1);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      
      s = 2.0 * posi - posj - posk;
      p = posi - (m_dish * 0.5) * s / math::abs(s);
      break;
      
    case 9: // (CH3)3-group (one psuedosite)
      assert(m_atom.size()>1);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      
      s = posi - posj;
      p = posi +  (m_disc + m_dish / 3.0) / (3 * math::abs(s)) * s;
      break;

    case 10: // cog
      {
	assert(m_atom.size() > 0);
	math::Vec cog(0.0, 0.0, 0.0);
	for(unsigned int i=1; i<m_atom.size(); ++i){
	  math::Vec v;
	  periodicity.nearest_image(position(m_atom[i]), position(m_atom[0]), v);
	  cog += v;
	}
	cog /= m_atom.size();
	p = cog + position(m_atom[0]);
	break;
      }
      
    default:
      io::messages.add("Virual Atom", "wrong type", io::message::critical);
      p = math::Vec(0,0,0);
  }
}

math::Vec util::Virtual_Atom::pos(configuration::Configuration & conf)const
{
  math::Vec p;
  SPLIT_BOUNDARY(_pos, conf.current().pos, conf.current().box, p);
  return p;
}

template<math::boundary_enum B>
void util::Virtual_Atom::_force
(
 math::VArray const & position,
 math::Box const & box,
 math::Vec const & f,
 math::VArray & force
 )const
{

  math::Vec posi, posj, posk, posl;
  math::Periodicity<B> periodicity(box);
  
  posi = position(m_atom[0]);

  math::Vec s,t,a,b, calc1, calc2, calc3, calch;
  double abs_s, abs_t;
  double m_c; //multiplication constant
  
  switch(m_type){
    
    case 0: // explicit atom
    case 7: // rotating ring
      assert(m_atom.size()>0);
      force(m_atom[0])+=f;
      break;
    case 1: // CH1
      assert(m_atom.size()>3);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;
      periodicity.nearest_image(position(m_atom[3]), posi, posl);
      posl += posi;

      s = 3.0 * posi - posj - posk - posl;
      abs_s = math::abs(s);
      m_c=3*m_dish;
      
      calc1= math::Vec(m_c*(abs_s*abs_s - s(0)*s(0)),
		       -m_c*s(1)*s(0), 
		       -m_c*s(2)*s(0))/(abs_s*abs_s*abs_s) + math::Vec(1,0,0);
      calc2= math::Vec(-m_c*s(1)*s(0),
		       m_c*(abs_s*abs_s - s(1)*s(1)),
		       -m_c*s(2)*s(1))/(abs_s*abs_s*abs_s) + math::Vec(0,1,0);
      calc3= math::Vec(-m_c*s(2)*s(0),
		       -m_c*s(2)*s(1),
		       m_c*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s) + math::Vec(0,0,1) ;     
      force(m_atom[0]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      
      calc1=m_c/3*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			    s(1)*s(0), 
			    s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2=m_c/3*math::Vec(s(1)*s(0),
			    -(abs_s*abs_s - s(1)*s(1)),
			    s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3=m_c/3*math::Vec(s(2)*s(0),
			    s(2)*s(1),
			    -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      force(m_atom[2]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      force(m_atom[3]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      break;

    case 2: // aromatic H
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;

      s = 2.0 * posi - posj - posk;
      abs_s = math::abs(s);
      
      calc1= math::Vec(2*m_dish*(abs_s*abs_s - s(0)*s(0)),
		       -2*m_dish*s(1)*s(0), 
		       -2*m_dish*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      calc2= math::Vec(-2*m_dish*s(1)*s(0),
		       2*m_dish*(abs_s*abs_s - s(1)*s(1)),
		       -2*m_dish*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      calc3= math::Vec(-2*m_dish*s(2)*s(0),
		       -2*m_dish*s(2)*s(1),
		       2*m_dish*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);     
      force(m_atom[0])=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
     

      calc1=m_dish*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			     s(1)*s(0), 
			     s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2=m_dish*math::Vec(s(1)*s(0),
			     -(abs_s*abs_s - s(1)*s(1)),
			     s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3=m_dish*math::Vec(s(2)*s(0),
			     s(2)*s(1),
			     -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      force(m_atom[2])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));

      break;
    case 3: // non-stereospecific CH2
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;

      s = 2.0 * posi - posj - posk;
      abs_s = math::abs(s);
      m_c=TETHCO*m_dish;
     
      calc1= math::Vec(2*m_c*(abs_s*abs_s - s(0)*s(0)),
		       -2*m_c*s(1)*s(0), 
		       -2*m_c*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      calc2= math::Vec(-2*m_c*s(1)*s(0),
		       2*m_c*(abs_s*abs_s - s(1)*s(1)),
		       -2*m_c*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      calc3= math::Vec(-2*m_c*s(2)*s(0),
		       -2*m_c*s(2)*s(1),
		       2*m_c*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);     
      force(m_atom[0])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      
      calc1= m_c*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			   s(1)*s(0), 
			   s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2= m_c*math::Vec(s(1)*s(0),
			   -(abs_s*abs_s - s(1)*s(1)),
			   s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3= m_c*math::Vec(s(2)*s(0),
			   s(2)*s(1),
			   -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      force(m_atom[2]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      break;
      
   case 4: // stereospecific CH2
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;
     
      DEBUG(8, "FORCE REDISTRIBUTION: case 4!!!");
      
      double m_c_2, abs_a, abs_b;
      
      s = 2.0 * posi - posj - posk;
      abs_s = math::abs(s);
      DEBUG(10, "\ts = " << math::v2s(s));
      
      t = math::cross(posi - posj, posi - posk);
      abs_t = math::abs(t);
      DEBUG(10, "\tq = " << math::v2s(t));
      
      b = posk- posj;
      abs_b = math::abs(b);
      DEBUG(10, "\tb = " << math::v2s(b));
      
      a = math::cross(t,b);
      abs_a = math::abs(a) ;
      DEBUG(10, "\ta = " << math::v2s(a));
      
      m_c = TETHCO*m_dish;
      m_c_2 = TETHSI*m_dish;
      
      calc1= math::Vec(2*m_c*(abs_s*abs_s - s(0)*s(0)),
		       -2*m_c*s(1)*s(0), 
		       -2*m_c*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      
      calc2= math::Vec(-2*m_c*s(1)*s(0),
		       2*m_c*(abs_s*abs_s - s(1)*s(1)),
		       -2*m_c*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      
      calc3= math::Vec(-2*m_c*s(2)*s(0),
		       -2*m_c*s(2)*s(1),
		       2*m_c*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);
      
      DEBUG(10, "A = " << math::v2s(calc1));
      DEBUG(10, "A = " << math::v2s(calc2));
      DEBUG(10, "A = " << math::v2s(calc3));
      
      calch= -m_c_2*math::Vec(t(0)*a(0), 
			      t(1)*a(0), 
			      t(2)*a(0))/(abs_t*abs_t*abs_t);
      DEBUG(10, "B = " << math::v2s(calch));
      
      calc1+=calch;     
      
      calch= -m_c_2*math::Vec(t(0)*a(1),
			      t(1)*a(1),
			      t(2)*a(1))/(abs_t*abs_t*abs_t);
      DEBUG(10, "B = " << math::v2s(calch));
      calc2+=calch;    
      
      calch= -m_c_2*math::Vec(t(0)*a(2),
			      t(1)*a(2),
			      t(2)*a(2))/(abs_t*abs_t*abs_t);
      DEBUG(10, "B = " << math::v2s(calch));
      calc3+=calch;
      
      calch = m_c_2*math::Vec(0, 
			      b(2), 
			      -b(1))/(abs_t);
      DEBUG(10, "C = " << math::v2s(calch));
      calc1+=calch;
      
      calch= m_c_2*math::Vec(-b(2),
			     0,
			     b(0))/(abs_t);
      DEBUG(10, "C = " << math::v2s(calch));
      calc2+=calch; 
      
      calch= m_c_2*math::Vec(b(1),
			     b(0),
			     0)/(abs_t);
      DEBUG(10, "C = " << math::v2s(calch));
      calc3+=calch;
      
      DEBUG(10, "drn/dri = " << math::v2s(calc1));
      DEBUG(10, "drn/dri = " << math::v2s(calc2));
      DEBUG(10, "drn/dri = " << math::v2s(calc3));
      
      force(m_atom[0]) += math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      DEBUG(10, "f(i) = " << math::v2s(math::Vec(math::dot(calc1,f),
						 math::dot(calc2,f),
						 math::dot(calc3,f))));
      
      calc1= m_c*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			   s(1)*s(0), 
			   s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2= m_c*math::Vec(s(1)*s(0),
			   -(abs_s*abs_s - s(1)*s(1)),
			   s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3= m_c*math::Vec(s(2)*s(0),
			   s(2)*s(1),
			   -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      
      DEBUG(10, "D = " << math::v2s(calc1));
      DEBUG(10, "D = " << math::v2s(calc2));
      DEBUG(10, "D = " << math::v2s(calc3));
      
      b = posi- posk;
      abs_b =  math::abs(b);
      a = math::cross(t,b);
      abs_a = math::abs(a) ;
      
      calch= -m_c_2*math::Vec(t(0)*a(0), 
			      t(1)*a(0), 
			      t(2)*a(0))/(abs_t*abs_t*abs_t);
      DEBUG(10, "E = " << math::v2s(calch));
      calc1+=calch;
      calch= -m_c_2*math::Vec(t(0)*a(1),
			      t(1)*a(1),
			      t(2)*a(1))/(abs_t*abs_t*abs_t); 
      DEBUG(10, "E = " << math::v2s(calch));
      calc2+=calch; 
      calch= -m_c_2*math::Vec(t(0)*a(2),
			      t(1)*a(2),
			      t(2)*a(2))/(abs_t*abs_t*abs_t);
      DEBUG(10, "E = " << math::v2s(calch));
      calc3+=calch;
      
      calch= m_c_2*math::Vec(0, 
			     b(2), 
			     -b(1))/(abs_t);
      DEBUG(10, "F = " << math::v2s(calch));
      calc1+=calch;
      calch= m_c_2*math::Vec(-b(2),
			     0,
			     b(0))/(abs_t);
      DEBUG(10, "F = " << math::v2s(calch));
      calc2+=calch;
      calch= m_c_2*math::Vec(b(1),
			     -b(0),
			     0)/(abs_t);
      DEBUG(10, "F = " << math::v2s(calch));
      calc3+=calch;
      force(m_atom[1])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      DEBUG(10, "f(j) = " << math::v2s(math::Vec(math::dot(calc1,f),
						 math::dot(calc2,f),
						 math::dot(calc3,f))));
      
      calc1= m_c*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			   s(1)*s(0), 
			   s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2= m_c*math::Vec(s(1)*s(0),
			   -(abs_s*abs_s - s(1)*s(1)),
			   s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3= m_c*math::Vec(s(2)*s(0),
			   s(2)*s(1),
			   -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      
      DEBUG(10, "D = " << math::v2s(calc1));
      DEBUG(10, "D = " << math::v2s(calc2));
      DEBUG(10, "D = " << math::v2s(calc3));
      
      b = posj - posi;
      abs_b = math::abs(b) ;
      a = math::cross(t,b );
      abs_a = math::abs(a) ;
      
      calch= -m_c_2*math::Vec(t(0)*a(0), 
			      t(1)*a(0), 
			      t(2)*a(0))/(abs_t*abs_t*abs_t) ;
      DEBUG(10, "G = " << math::v2s(calch));
      calc1+=calch;
      calch = -m_c_2*math::Vec(t(0)*a(1),
			       t(1)*a(1),
			       t(2)*a(1))/(abs_t*abs_t*abs_t);
      DEBUG(10, "G = " << math::v2s(calch));
      calc2+=calch;
      calch= -m_c_2*math::Vec(t(0)*a(2),
			      t(1)*a(2),
			      t(2)*a(2))/(abs_t*abs_t*abs_t);
      DEBUG(10, "G = " << math::v2s(calch));
      calc3+=calch;			
      
      calch= m_c_2*math::Vec(0, 
			     b(2), 
			     -b(1))/(abs_t);
      DEBUG(10, "H = " << math::v2s(calch));
      calc1+=calch;
      calch= m_c_2*math::Vec(-b(2),
			     0,
			     b(0))/(abs_t);
      DEBUG(10, "H = " << math::v2s(calch));
      calc2+=calch;
      calch= m_c_2*math::Vec(b(1),
			     -b(0),
			     0)/(abs_t) ;
      DEBUG(10, "H = " << math::v2s(calch));
      calc3+=calch;			
      
      force(m_atom[2])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      DEBUG(10, "f(k) = " << math::v2s(math::Vec(math::dot(calc1,f),
						 math::dot(calc2,f),
						 math::dot(calc3,f))));
      break;
      
    case 5: // CH3
      assert(m_atom.size()>1);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;

      s =  posi - posj;
      abs_s = math::abs(s);
      m_c= m_dish/3;
      
      calc1= math::Vec(m_c*(abs_s*abs_s - s(0)*s(0)),
		       -m_c*s(1)*s(0), 
		       -m_c*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      calc2= math::Vec(-m_c*s(1)*s(0),
		       m_c*(abs_s*abs_s - s(1)*s(1)),
		       -m_c*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      calc3= math::Vec(-m_c*s(2)*s(0),
		       -m_c*s(2)*s(1),
		       m_c*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);     
      force(m_atom[0])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      calc1= m_c*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			   s(1)*s(0), 
			   s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2= m_c*math::Vec(s(1)*s(0),
			   -(abs_s*abs_s - s(1)*s(1)),
			   s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3= m_c*math::Vec(s(2)*s(0),
			   s(2)*s(1),
			   -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      break;
      
    case 6: // non-stereospecific CH3 (Leu, Val)
      assert(m_atom.size()>2);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;

      s = 2.0 * posi - posj - posk;
      abs_s=math::abs(s);    
      m_c=-TETHCO* (m_disc + m_dish / 3.0);
      
      calc1= math::Vec(2*m_c*(abs_s*abs_s - s(0)*s(0)),
		       -2*m_c*s(1)*s(0), 
		       -2*m_c*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      calc2= math::Vec(-2*m_c*s(1)*s(0),
		       2*m_c*(abs_s*abs_s - s(1)*s(1)),
		       -2*m_c*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      calc3= math::Vec(-2*m_c*s(2)*s(0),
		       -2*m_c*s(2)*s(1),
		       2*m_c*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);     
      force(m_atom[0])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      calc1= m_c*math::Vec(-m_dish*(abs_s*abs_s - s(0)*s(0)),
			   m_dish*s(1)*s(0), 
			   m_dish*s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2= m_c*math::Vec(m_dish*s(1)*s(0),
			   -m_dish*(abs_s*abs_s - s(1)*s(1)),
			   m_dish*s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3= m_c*math::Vec(m_dish*s(2)*s(0),
			   m_dish*s(2)*s(1),
			   -m_dish*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      force(m_atom[2])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      break;
    case 8: // NH2-group (one pseudosite)
      assert(m_atom.size()>1);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;
      periodicity.nearest_image(position(m_atom[2]), posi, posk);
      posk += posi;

      s = 2.0 * posi - posj - posk;
      abs_s=math::abs(s);
     
      calc1= math::Vec(-m_dish*(abs_s*abs_s - s(0)*s(0)),
		       m_dish*s(1)*s(0), 
		       m_dish*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      calc2= math::Vec(m_dish*s(1)*s(0),
		       -m_dish*(abs_s*abs_s - s(1)*s(1)),
		       m_dish*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      calc3= math::Vec(m_dish*s(2)*s(0),
		       m_dish*s(2)*s(1),
		       -m_dish*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);     
      force(m_atom[0])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      calc1=-0.5*m_dish*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
				  s(1)*s(0), 
				  s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2=-0.5*m_dish*math::Vec(s(1)*s(0),
				  -(abs_s*abs_s - s(1)*s(1)),
				  s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3=-0.5*m_dish*math::Vec(s(2)*s(0),
				  s(2)*s(1),
				  -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      force(m_atom[2])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      break;
      
    case 9: // (CH3)3-group (one psuedosite)
      assert(m_atom.size()>1);
      periodicity.nearest_image(position(m_atom[1]), posi, posj);
      posj += posi;

      s =  posi - posj;
      abs_s = math::abs(s);
      m_c=  ( m_disc + m_dish/3.0 )/ 3;
      
      calc1= math::Vec(m_c*(abs_s*abs_s - s(0)*s(0)),
		       -m_c*s(1)*s(0), 
		       -m_c*s(2)*s(0))/(abs_s*abs_s*abs_s)+math::Vec(1,0,0);
      calc2= math::Vec(-m_c*s(1)*s(0),
		       m_c*(abs_s*abs_s - s(1)*s(1)),
		       -m_c*s(2)*s(1))/(abs_s*abs_s*abs_s)+math::Vec(0,1,0);
      calc3= math::Vec(-m_c*s(2)*s(0),
		       -m_c*s(2)*s(1),
		       m_c*(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s)+math::Vec(0,0,1);     
      force(m_atom[0])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      
      calc1= m_c*math::Vec(-(abs_s*abs_s - s(0)*s(0)),
			   s(1)*s(0), 
			   s(2)*s(0))/(abs_s*abs_s*abs_s);
      calc2= m_c*math::Vec(s(1)*s(0),
			   -(abs_s*abs_s - s(1)*s(1)),
			   s(2)*s(1))/(abs_s*abs_s*abs_s);
      calc3= m_c*math::Vec(s(2)*s(0),
			   s(2)*s(1),
			   -(abs_s*abs_s - s(2)*s(2)))/(abs_s*abs_s*abs_s);
      force(m_atom[1])+=math::Vec(math::dot(calc1,f),math::dot(calc2,f),math::dot(calc3,f));
      break;

    case 10: // cog
      assert(m_atom.size() > 0);
      {
	for(unsigned int i=0; i<m_atom.size(); ++i){
	  force(m_atom[i]) += f / m_atom.size();
	}
	break;
      }
      
    default:
      std::cerr <<"Type not implemented";
      assert(false);
      break;
  }
}

void util::Virtual_Atom::force
(
 configuration::Configuration & conf,
 math::Vec const f
)const
{
  SPLIT_BOUNDARY(_force, conf.current().pos, conf.current().box, f, conf.current().force);
}
