/* Burau Search version 1.2
 * Search for a whisker whose pairing with the standard fork
 * is zero, thus giving an element of ker(Burau4).
 * This implements the simplest algorithm.
 *
 * Changes from version 1.1:
 *
 * Increased FREQUENCY to 1234567
 * Changed every exit to return
 * Added one_root function to quickly check if q=1 is a root
 *   and the variables specify a single whisker.
 *   eg. 3,0,0,1,3 = 2,0,0,0,2 union a loop.
 */

#include<stdio.h>
#include<stdlib.h>

#define LOG "burau.log"
#define PROJECT "burau.proj"
#define FREQUENCY 1234567
/* the frequency with which it outputs its progress to LOG */

/* values for the test_run */
#define START_TEST 2
#define STOP_TEST 50
#define Q_TEST 3

int pairing(int a,int b,int c,int d,int e,int q)
/* return value of the intersection pairing of
 * the whisker defined by a,b,c,d,e and a standard fork
 * evaluated at q
 */
/* Whisker defined by a,b,c,d,e is as follows:
 * Arrange top row of three punctures, bottom row of two punctures.
 * The top middle puncture is the "puncture at infinity".
 * The fork joins the bottom two punctures by a horizontal line.
 * The whisker goes from the puncture at infinity to the top right puncture.
 * Variables a,...,e specify the number of parallel strands 
 * around punctures, reading left to right, top to bottom.
 */
{
  int bl,start,cl,end, el,er;
  /* bl = first strand in left foot of hump b, */
  /* start = starting strand (in the middle of hump b), */
  /* end = ending strand (in the middle of hump c).*/

  int suma,sumb,sumc,sumd,sume;
  /* sum of first and last strands in hump a,b,c,d,e */

  int x;
  /* current x coordinate */

  int poly = 0;
  /* value of pairing at q */
  int monomial = 1;
  int q4;

  q4 = q*q*q*q;

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    /* go down&thru */
    if(x < el)
    {
      /* in hump d */
      if(x < d)
      {
        monomial *= q;
        poly += monomial;
      }
      else
      {
        poly -= monomial;
        poly *= q;
      }
      x = sumd - x;
    }
    else
    {
      /* in hump e */
      if(x < er)
      {
        poly -= monomial;
        monomial *= q;
      }
      else
      {
        poly *= q;
        poly += monomial;
      }
      x = sume - x;
    }

    /* go over */
    if(x < cl)
    {
      if(x < bl)
      {
        /* in hump a */
        if(x < a)
          poly *= q;
        else
          monomial *= q;
        x = suma-x;
      }
      else
      {
        /* in hump b */
        if(x < start)
          monomial *= q4;
        else
          poly *= q4;
        x = sumb-x;
      }
    }
    else
    {
      /* in hump c */
      if(x < end)
        poly *= q;
      else if(x > end)
        monomial *= q;
      else
        return(poly);
      x = sumc-x;
    }
  }
}

int positive_mod(int x,int p)
/* return x modulo p in the range 0,...,p-1 */
{
  x %= p;
  if(x < 0)
    x += p;
  return(x);
}

int multiply_mod(int x,int y,int p)
/* return x*y modulo p */
{
  long long xx = positive_mod(x,p);
  long long yy = positive_mod(y,p);
  return((int)((xx*yy)%p));
}

int pairing_mod(int a,int b,int c,int d,int e,int q,int p)
/* return value of the intersection pairing modulo p at q */
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int poly = 0;
  int monomial = 1;
  int q_mod = positive_mod(q,p);
  int q4 = multiply_mod(
    multiply_mod(q_mod,q_mod,p),
    multiply_mod(q_mod,q_mod,p),
    p
  );

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    /* go down&thru */
    if(x < el)
    {
      /* in hump d */
      if(x < d)
      {
        monomial = multiply_mod(monomial,q_mod,p);
        poly = positive_mod(poly + monomial,p);
      }
      else
      {
        poly = positive_mod(poly - monomial,p);
        poly = multiply_mod(poly,q_mod,p);
      }
      x = sumd - x;
    }
    else
    {
      /* in hump e */
      if(x < er)
      {
        poly = positive_mod(poly - monomial,p);
        monomial = multiply_mod(monomial,q_mod,p);
      }
      else
      {
        poly = multiply_mod(poly,q_mod,p);
        poly = positive_mod(poly + monomial,p);
      }
      x = sume - x;
    }

    /* go over */
    if(x < cl)
    {
      if(x < bl)
      {
        /* in hump a */
        if(x < a)
          poly = multiply_mod(poly,q_mod,p);
        else
          monomial = multiply_mod(monomial,q_mod,p);
        x = suma-x;
      }
      else
      {
        /* in hump b */
        if(x < start)
          monomial = multiply_mod(monomial,q4,p);
        else
          poly = multiply_mod(poly,q4,p);
        x = sumb-x;
      }
    }
    else
    {
      /* in hump c */
      if(x < end)
        poly = multiply_mod(poly,q_mod,p);
      else if(x > end)
        monomial = multiply_mod(monomial,q_mod,p);
      else
        return(poly);
      x = sumc-x;
    }
  }
}

int single_whisker(int a,int b,int c,int d,int e)
/* return 1 iff a,...,e represents a single whisker */
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int togo;

  togo = d+e;

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    /* go down&thru */
    if(x < el)
      x = sumd - x;
    else
      x = sume - x;
    togo--;

    /* go over */
    if(x < cl)
    {
      if(x < bl)
        x = suma-x;
      else
        x = sumb-x;
    }
    else
    {
      /* in hump c */
      if(x == end)
        return(!togo);
      x = sumc-x;
    }
  }
}

int one_root(int a,int b,int c,int d,int e)
/* return 1 iff one a,...,e represents a single whisker,
 * AND one is a root of the pairing.
 */
{
  int bl,start,cl,end, el,er;
  /* bl = first strand in left foot of hump b, */
  /* start = starting strand (in the middle of hump b), */
  /* end = ending strand (in the middle of hump c).*/

  int suma,sumb,sumc,sumd,sume;
  /* sum of first and last strands in hump a,b,c,d,e */

  int x;
  /* current x coordinate */

  int poly = 0;
  /* value of pairing at q=1 */

  int togo;
  /* number of crossings to go */
  togo = d+e;

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    /* go down&thru */
    if(x < el)
    {
      /* in hump d */
      if(x < d)
        poly++;
      else
        poly--;
      x = sumd - x;
    }
    else
    {
      /* in hump e */
      if(x < er)
        poly--;
      else
        poly++;
      x = sume - x;
    }
    togo--;
    if(poly > togo)
      return(0);
    if(poly < -togo)
      return(0);

    /* go over */
    if(x < cl)
    {
      if(x < bl)
        x = suma-x;
      else
        x = sumb-x;
    }
    else
    {
      /* in hump c */
      if(x == end)
        return(!togo);
      x = sumc-x;
    }
  }
}

int possible(int a,int b,int c,int d,int e)
/* return 1 if the pairing of this whisker and a standard fork
 * evaluates zero at several q, so is probably zero
 */
{
  if(!one_root(a,b,c,d,e))
    return(0);
  if(pairing(a,b,c,d,e,3))
    return(0);
  if(pairing(a,b,c,d,e,5))
    return(0);
  if(pairing(a,b,c,d,e,7))
    return(0);
  if(pairing(a,b,c,d,e,11))
    return(0);
  return(1);
}

int possible_mod_p(int a,int b,int c,int d,int e,int p)
/* return 1 if this single whisker has pairing zero at every q in F_p
 */
{
  int q;
  if(!single_whisker(a,b,c,d,e))
    return(0);
  for(q = 0; q < p; q++)
    if(pairing_mod(a,b,c,d,e,q,p))
      return(0);
  return(1);
}

void increment(int* a,int* b,int* c,int* d,int* e)
/* move on to the next whisker */
{
  if(*e)
  {
    (*d)++;
    (*e)--;
    return;
  }
  if(*c)
  {
    (*b)++;
    (*c)--;
  }
  else if(*b)
  {
    (*a)++;
    *c = *b-1;
    *b = 0;
  }
  else
  {
    *c = *a+2;
    *a = 0;
  }
  *e = *a+*b+*c+1;
  *d = 0;
}

int ok_computer(void)
/* Test to see if this machine-dependent code works on this machine */
{
  unsigned int i=2;
  unsigned int third=1;
  while(i)
  {
    third += i;
    i *= 4;
  }
  /* third should now be binary ...10101011 */
  return(third*3u == 1u);
}

int test_run(FILE* fp)
/* Find an example with a root at q=Q_TEST 
 * in the range START_TEST to STOP_TEST (if one exists).
 * This gives a check that the program is working properly.
 */
{
  int a=0;
  int b=0;
  int c=START_TEST-1;
  int d=0;
  int e=START_TEST;

  while(d+e <= STOP_TEST)
  {
    if(pairing(a,b,c,d,e,Q_TEST) == 0)
    {
      fprintf(fp,"%d,%d,%d,%d,%d\n",a,b,c,d,e);
      fprintf(fp,"Enter this at ");
      fprintf(fp,"http://www.ms.unimelb.edu.au/~bigelow/member.html\n");
      fclose(fp);
      return(0);
    }
    increment(&a,&b,&c,&d,&e);
  }
  return(1);
}

int main(int argc,char** argv)
{
  FILE* fp;

  int count = 2;

  int a,b,c,d,e;
  int start,stop;
  int modulus = 0;
  int argi;

  for(argi = 1; argi < argc; argi++)
  {
    int parsed = atoi(argv[argi]);
    if(parsed > 1)
    {
      modulus = parsed;
      break;
    }
  }

  if(!ok_computer())
  {
    fp = fopen(LOG,"a");
    fprintf(fp,"This machine dependent code won't work on this machine.\n");
    printf("This machine dependent code won't work on this machine.\n");
    fclose(fp);
    return(1);
  }

  fp = fopen(PROJECT,"r");
  if(fp == NULL)
  /* If no project given, do a test run */
  {
    fp = fopen(LOG,"a");
    return(test_run(fp));
  }

  fscanf(fp,"From %d to %d",&start,&stop);
  fclose(fp);
  start &= ~1; /*start must be an even number*/

  fp = fopen(LOG,"r");
  if(fp == NULL)
  {
    a=0;
    b=0;
    c=start-1;
    d=0;
    e=start;
  }
  else
  {
    fscanf(fp,"%d,%d,%d,%d,%d",&a,&b,&c,&d,&e);
    fclose(fp);
  }

  if(d+e > stop)
  {
    printf("\nData in burau.log occurs after project in burau.proj\n");
    return(1);
  }

  while(d+e <= stop)
  {
    if(
      (modulus == 0 && possible(a,b,c,d,e))
      || (modulus != 0 && possible_mod_p(a,b,c,d,e,modulus))
    )
    {
      fp = fopen(LOG,"w");
      fprintf(fp,"%d,%d,%d,%d,%d\n",a,b,c,d,e);
      if(modulus == 0)
        fprintf(fp,"Example found!!!\n");
      else
        fprintf(fp,"Example found modulo %d!!!\n",modulus);
      fprintf(fp,"Please email this file to bigelow@unimelb.edu.au\n");
      fclose(fp);
      return(0);
    }
    if(--count == 0)
    {
      count = FREQUENCY;
      fp = fopen(LOG,"w");
      fprintf(fp,"%d,%d,%d,%d,%d\n",a,b,c,d,e);
      fclose(fp);
    }
    if(d+e < start)
    {
      a=0;
      b=0;
      c=start-1;
      d=0;
      e=start;
    }
    /* move to next curve */
    else
    {
      increment(&a,&b,&c,&d,&e);
    }
  }
  fp = fopen(LOG,"w");
  fprintf(fp,"%d,%d,%d,%d,%d\n",a,b,c,d,e);
  fprintf(fp,"Finished checking from %d to %d\n",start,stop);
  fclose(fp);
  return(0);
}
