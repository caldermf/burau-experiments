#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void ensure_capacity(int** coeffs,int* cap,int needed)
{
  int old_cap;
  int* new_coeffs;

  if(needed <= *cap)
    return;

  old_cap = *cap;
  while(*cap < needed)
    *cap *= 2;

  new_coeffs = calloc((size_t)(*cap),sizeof(int));
  memcpy(new_coeffs,*coeffs,(size_t)old_cap*sizeof(int));
  free(*coeffs);
  *coeffs = new_coeffs;
}

void shift_poly(int** coeffs,int* len,int* cap,int amount)
{
  int i;
  if(amount == 0 || *len == 0)
    return;

  ensure_capacity(coeffs,cap,*len + amount);
  for(i = *len - 1; i >= 0; i--)
    (*coeffs)[i + amount] = (*coeffs)[i];
  for(i = 0; i < amount; i++)
    (*coeffs)[i] = 0;
  *len += amount;
}

void add_monomial(int** coeffs,int* len,int* cap,int exp,int delta,int p)
{
  int needed = exp + 1;
  ensure_capacity(coeffs,cap,needed);
  while(*len < needed)
  {
    (*coeffs)[*len] = 0;
    (*len)++;
  }
  (*coeffs)[exp] = positive_mod((*coeffs)[exp] + delta,p);
}

int exact_zero_mod_p(int a,int b,int c,int d,int e,int p)
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int mon_exp = 0;
  int len = 0;
  int cap = 64;
  int* coeffs = calloc((size_t)cap,sizeof(int));
  int i;

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
        mon_exp++;
        add_monomial(&coeffs,&len,&cap,mon_exp,1,p);
      }
      else
      {
        add_monomial(&coeffs,&len,&cap,mon_exp,-1,p);
        shift_poly(&coeffs,&len,&cap,1);
      }
      x = sumd - x;
    }
    else
    {
      if(x < er)
      {
        add_monomial(&coeffs,&len,&cap,mon_exp,-1,p);
        mon_exp++;
      }
      else
      {
        shift_poly(&coeffs,&len,&cap,1);
        add_monomial(&coeffs,&len,&cap,mon_exp,1,p);
      }
      x = sume - x;
    }

    if(x < cl)
    {
      if(x < bl)
      {
        if(x < a)
          shift_poly(&coeffs,&len,&cap,1);
        else
          mon_exp++;
        x = suma-x;
      }
      else
      {
        if(x < start)
          mon_exp += 4;
        else
          shift_poly(&coeffs,&len,&cap,4);
        x = sumb-x;
      }
    }
    else
    {
      if(x < end)
        shift_poly(&coeffs,&len,&cap,1);
      else if(x > end)
        mon_exp++;
      else
      {
        for(i = 0; i < len; i++)
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
  int p = 3;
  int stop = 400;
  int progress = 1000000;
  int a,b,c,d,e;
  long checked = 0;

  if(argc >= 2)
    stop = atoi(argv[1]);
  if(argc >= 3)
    progress = atoi(argv[2]);
  if(argc >= 4)
    p = atoi(argv[3]);

  a = 0;
  b = 0;
  c = 1;
  d = 0;
  e = 2;

  while(d+e <= stop)
  {
    if(single_whisker(a,b,c,d,e) && exact_zero_mod_p(a,b,c,d,e,p))
    {
      printf("HIT p=%d stop=%d tuple=(%d,%d,%d,%d,%d) checked=%ld\n",
        p,stop,a,b,c,d,e,checked);
      return(0);
    }

    checked++;
    if(progress > 0 && checked % progress == 0)
    {
      printf("progress checked=%ld current=(%d,%d,%d,%d,%d) level=%d\n",
        checked,a,b,c,d,e,d+e);
      fflush(stdout);
    }
    increment(&a,&b,&c,&d,&e);
  }

  printf("NO_HIT p=%d stop=%d checked=%ld final_level=%d\n",
    p,stop,checked,d+e);
  return(0);
}
