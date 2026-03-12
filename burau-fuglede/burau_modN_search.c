#include <stdio.h>
#include <stdlib.h>

int positive_mod(int x,int p)
{
  x %= p;
  if(x < 0)
    x += p;
  return(x);
}

void increment(int* a,int* b,int* c,int* d,int* e)
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

int single_whisker(int a,int b,int c,int d,int e)
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int togo = d+e;

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
    if(x < el)
      x = sumd - x;
    else
      x = sume - x;
    togo--;

    if(x < cl)
    {
      if(x < bl)
        x = suma-x;
      else
        x = sumb-x;
    }
    else
    {
      if(x == end)
        return(!togo);
      x = sumc-x;
    }
  }
}

void shift_poly(int* coeffs,int n,int amount)
{
  int* tmp = calloc((size_t)n,sizeof(int));
  int i;
  for(i = 0; i < n; i++)
    tmp[(i + amount) % n] = coeffs[i];
  for(i = 0; i < n; i++)
    coeffs[i] = tmp[i];
  free(tmp);
}

int zero_mod_tN_minus_1(int a,int b,int c,int d,int e,int p,int n)
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int monomial = 0;
  int i;
  int* coeffs = calloc((size_t)n,sizeof(int));

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
    if(x < el)
    {
      if(x < d)
      {
        monomial = (monomial + 1) % n;
        coeffs[monomial] = positive_mod(coeffs[monomial] + 1,p);
      }
      else
      {
        coeffs[monomial] = positive_mod(coeffs[monomial] - 1,p);
        shift_poly(coeffs,n,1);
      }
      x = sumd - x;
    }
    else
    {
      if(x < er)
      {
        coeffs[monomial] = positive_mod(coeffs[monomial] - 1,p);
        monomial = (monomial + 1) % n;
      }
      else
      {
        shift_poly(coeffs,n,1);
        coeffs[monomial] = positive_mod(coeffs[monomial] + 1,p);
      }
      x = sume - x;
    }

    if(x < cl)
    {
      if(x < bl)
      {
        if(x < a)
          shift_poly(coeffs,n,1);
        else
          monomial = (monomial + 1) % n;
        x = suma-x;
      }
      else
      {
        if(x < start)
          monomial = (monomial + 4) % n;
        else
          shift_poly(coeffs,n,4 % n);
        x = sumb-x;
      }
    }
    else
    {
      if(x < end)
        shift_poly(coeffs,n,1);
      else if(x > end)
        monomial = (monomial + 1) % n;
      else
      {
        for(i = 0; i < n; i++)
          if(coeffs[i] != 0)
          {
            free(coeffs);
            return(0);
          }
        free(coeffs);
        return(1);
      }
      x = sumc-x;
    }
  }
}

int main(int argc,char** argv)
{
  int p;
  int start = 2;
  int stop = 200;
  int n;
  int a,b,c,d,e;
  long checked = 0;

  if(argc != 2)
  {
    fprintf(stderr,"Usage: %s p\\n",argv[0]);
    return(1);
  }

  p = atoi(argv[1]);
  if(p <= 1)
  {
    fprintf(stderr,"p must be > 1\\n");
    return(1);
  }

  n = p*p - 1;
  start &= ~1;
  a = 0;
  b = 0;
  c = start-1;
  d = 0;
  e = start;

  while(d+e <= stop)
  {
    if(single_whisker(a,b,c,d,e) && zero_mod_tN_minus_1(a,b,c,d,e,p,n))
    {
      printf("p=%d N=%d tuple=(%d,%d,%d,%d,%d) checked=%ld\\n",
        p,n,a,b,c,d,e,checked);
      return(0);
    }
    checked++;
    increment(&a,&b,&c,&d,&e);
  }

  printf("p=%d N=%d no hit through stop=%d checked=%ld\\n",p,n,stop,checked);
  return(0);
}
